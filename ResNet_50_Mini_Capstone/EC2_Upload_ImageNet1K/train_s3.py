#!/usr/bin/env python3
# train_s3.py -- stall-resistant, S3-streaming aware training loop for ResNet-50

import argparse
import os
import sys
import time
import traceback
import itertools
import threading
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# boto3 for explicit S3 health checks & fallback listing (we still rely on utils.get_latest_s3_checkpoint)
import boto3
import botocore

# # Try to import NVML binding in a robust way (support both nvidia_smi and py3nvml)
_nvidia_module = None
try:
    import nvidia_smi as _nvidia_module  # preferred
except Exception:
    try:
        from py3nvml import py3nvml as _nvidia_module
    except Exception:
        _nvidia_module = None

# Local imports (assumes package layout)
from .config import Config
from .model import resnet50
from .data_loader import get_data_loaders
from .utils import (
    setup_logging,
    AverageMeter,
    ProgressMeter,
    accuracy,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_device,
    count_parameters,
    Timer,
    get_latest_s3_checkpoint
)


# ---------- NVML helpers (safe) ----------
_nvml_initialized = False
def get_gpu_utilization(device_id: int = 0) -> float:
    """Return GPU utilization percent or -1.0 if unavailable."""
    global _nvml_initialized, _nvidia_module
    if _nvidia_module is None:
        return -1.0
    try:
        if not _nvml_initialized:
            # two APIs differ slightly; both expose nvmlInit
            _nvidia_module.nvmlInit()
            _nvml_initialized = True
    except Exception:
        _nvml_initialized = False
        return -1.0

    try:
        handle = _nvidia_module.nvmlDeviceGetHandleByIndex(device_id)
        util = _nvidia_module.nvmlDeviceGetUtilizationRates(handle)
        # util.gpu is typically int
        return float(util.gpu)
    except Exception:
        return -1.0

def shutdown_nvml():
    global _nvml_initialized, _nvidia_module
    if _nvidia_module is None:
        return
    if _nvml_initialized:
        try:
            _nvidia_module.nvmlShutdown()
            _nvml_initialized = False
        except Exception:
            pass
# ---------- end NVML helpers ----------


# ---------- S3 helpers ----------
def s3_client_from_config(config):
    """Create a boto3 client with conservative timeouts / retries tuned for S3 streaming."""
    # Build botocore config
    botocore_cfg = botocore.config.Config(
        retries={"max_attempts": getattr(config, "s3_max_retries", 5), "mode": "standard"},
        connect_timeout=getattr(config, "s3_connect_timeout", 10),
        read_timeout=getattr(config, "s3_read_timeout", 60),
    )
    # Create client (region is optional)
    kwargs = {}
    if getattr(config, "s3_region", None):
        kwargs["region_name"] = config.s3_region
    return boto3.client("s3", config=botocore_cfg, **kwargs)


def check_s3_access(config, logger=None):
    """Quick lightweight S3 access check (list bucket keys for MaxKeys=1)."""
    try:
        s3 = s3_client_from_config(config)
        # If bucket is None, skip
        if not getattr(config, "s3_bucket", None):
            return False
        s3.list_objects_v2(Bucket=config.s3_bucket, Prefix=(config.s3_prefix_train or ""), MaxKeys=1)
        return True
    except Exception as e:
        if logger:
            logger.warning(f"S3 access check failed: {e}")
        return False
# ---------- end S3 helpers ----------


# ---------- Watchdog (thread) ----------
def start_watchdog(pbar, logger, timeout_sec=900, check_interval=60):
    """Start a background thread that warns if progress hasn't advanced in timeout_sec.
    pbar must be a tqdm instance.
    """
    if pbar is None:
        return None

    state = {"last_n": getattr(pbar, "n", -1), "last_time": time.time()}

    def _watch():
        while True:
            time.sleep(check_interval)
            try:
                current_n = getattr(pbar, "n", -1)
                if current_n == state["last_n"]:
                    # no progress since last check_interval
                    elapsed = time.time() - state["last_time"]
                    if elapsed >= timeout_sec:
                        logger.warning(f"⚠️ No progress on progress bar for {int(elapsed)}s (>= {timeout_sec}s). "
                                       "Possible DataLoader stall or S3 I/O issue.")
                        # reset timer so we don't spam
                        state["last_time"] = time.time()
                else:
                    state["last_n"] = current_n
                    state["last_time"] = time.time()
            except Exception:
                # be robust: don't let exceptions kill the watchdog thread
                pass

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t
# ---------- end Watchdog ----------


# ---------- Training / Validation functions ----------
def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    config,
    logger,
    scaler=None,
    scheduler=None,
    start_batch_idx=0,
    current_best_acc1=0.0,
    current_best_train_acc1=0.0
):
    batch_time = AverageMeter('Time', '6.3f')
    data_time = AverageMeter('Data', '6.3f')
    losses = AverageMeter('Loss', '.4e')
    top1 = AverageMeter('Acc@1', '6.2f')
    top5 = AverageMeter('Acc@5', '6.2f')

    model.train()

    if start_batch_idx > 0:
        logger.info(f"Resuming epoch {epoch+1} from batch {start_batch_idx+1}/{len(train_loader)}...")
        train_loader_iter = itertools.islice(train_loader, start_batch_idx, None)
    else:
        train_loader_iter = train_loader

    pbar = tqdm(
        enumerate(train_loader_iter, start=start_batch_idx),
        total=len(train_loader),
        initial=start_batch_idx,
        desc=f"Epoch {epoch+1}/{config.epochs}",
        leave=False, dynamic_ncols=True
    )

    # Start watchdog to warn if no progress
    watchdog = start_watchdog(pbar, logger, timeout_sec=getattr(config, "dataloader_timeout", 900))

    end = time.time()

    use_amp = getattr(config, "mixed_precision", False)
    use_bf16 = getattr(config, "use_bf16", False)
    device_type = 'cuda' if config.device.type == 'cuda' else 'cpu'

    if use_amp and device_type == 'cuda':
        if use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        autocast_ctx = torch.nullcontext()

    grad_accum_steps = getattr(config, "grad_accum_steps", 1)

    for i, (images, target) in pbar:
        current_data_time = time.time() - end
        data_time.update(current_data_time)

        # Move data to device
        if config.device.type == 'cuda' and getattr(config, "use_channels_last", True):
            images = images.to(config.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        else:
            images = images.to(config.device, non_blocking=True)
        target = target.to(config.device, non_blocking=True)

        with autocast_ctx:
            output = model(images)
            loss = criterion(output, target) / grad_accum_steps

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item() * grad_accum_steps, images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if (i % grad_accum_steps) == 0:
            optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Optionally update scheduler per step if OneCycleLR
            if scheduler is not None and getattr(config, "lr_scheduler", "").lower() == "onecycle":
                scheduler.step()

        current_batch_time = time.time() - end
        batch_time.update(current_batch_time)
        end = time.time()

        # GPU util (safe)
        gpu_util = get_gpu_utilization(config.device.index if getattr(config.device, "type", None) == "cuda" else 0)
        gpu_util_str = f'{gpu_util:.0f}%' if gpu_util >= 0 else 'N/A'

        # Try to get lr safely
        try:
            current_lr = optimizer.param_groups[0]['lr']
        except Exception:
            current_lr = getattr(config, "learning_rate", 0.0)

        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.2f}%',
            'Acc@5': f'{top5.avg:.2f}%',
            'LR': f'{current_lr:.6f}',
            'DataT': f'{data_time.val:.3f}s',
            'BatchT': f'{batch_time.val:.3f}s',
            'GPU%': gpu_util_str
        })

        if (i + 1) % getattr(config, "log_interval", 100) == 0:
            logger.info(
                f"Epoch [{epoch+1}] Batch [{i+1}/{len(train_loader)}] "
                f"Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}% "
                f"DataT: {data_time.val:.3f}s BatchT: {batch_time.val:.3f}s GPU%: {gpu_util_str}"
            )

        # Mid-epoch checkpoint
        if config.save_every_n_batches and config.save_every_n_batches > 0 and (i + 1) % config.save_every_n_batches == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_acc1': current_best_acc1,
                'best_train_acc1': current_best_train_acc1,
                'config': config
            }
            fname = f'checkpoint_s3_epoch_{epoch}_batch_{i+1}.pth'
            save_checkpoint(checkpoint, is_best=False, checkpoint_dir=config.checkpoint_dir,
                            filename=fname, s3_bucket=config.s3_checkpoint_bucket, s3_prefix=config.s3_checkpoint_prefix)
            logger.info(f"Saved mid-epoch checkpoint for epoch {epoch+1}, batch {i+1} (local and S3).")

    # stop watchdog automatically when function returns (daemon thread)
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, config, logger):
    batch_time = AverageMeter('Time', '6.3f')
    losses = AverageMeter('Loss', '.4e')
    top1 = AverageMeter('Acc@1', '6.2f')
    top5 = AverageMeter('Acc@5', '6.2f')

    model.eval()
    end = time.time()

    use_amp = getattr(config, "mixed_precision", False)
    use_bf16 = getattr(config, "use_bf16", False)
    device_type = 'cuda' if config.device.type == 'cuda' else 'cpu'

    if use_amp and device_type == 'cuda':
        if use_bf16:
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        amp_ctx = torch.nullcontext()

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)):
            if config.device.type == 'cuda' and getattr(config, "use_channels_last", True):
                images = images.to(config.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            else:
                images = images.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)

            with amp_ctx:
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    logger.info(f"Validation Results - Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%")
    return top1.avg, top5.avg, losses.avg
# ---------- end training/validation ----------


# ---------- Optimizer & scheduler helpers ----------
def get_optimizer(model, config):
    opt_name = getattr(config, "optimizer", "sgd").lower()
    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=config.learning_rate,
                         momentum=getattr(config, "momentum", 0.9),
                         weight_decay=getattr(config, "weight_decay", 1e-4),
                         nesterov=getattr(config, "nesterov", False))
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=getattr(config, "weight_decay", 1e-4))
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=getattr(config, "weight_decay", 1e-4))
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def get_lr_scheduler(optimizer, config):
    sched_name = getattr(config, "lr_scheduler", "step").lower()
    if sched_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=getattr(config, "lr_step_size", 30), gamma=getattr(config, "lr_gamma", 0.1))
    elif sched_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(config, "epochs", 90))
    elif sched_name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=getattr(config, "lr_milestones", [30,60,80]), gamma=getattr(config, "lr_gamma", 0.1))
    elif sched_name == 'onecycle':
        steps_per_epoch = getattr(config, "steps_per_epoch", None)
        if steps_per_epoch is None:
            raise ValueError("onecycle scheduler requires config.steps_per_epoch to be set")
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.epochs, steps_per_epoch=steps_per_epoch,
                                            pct_start=getattr(config, "onecycle_pct_start", 0.3),
                                            anneal_strategy=getattr(config, "onecycle_anneal", "cos"),
                                            final_div_factor=getattr(config, "onecycle_final_div_factor", 1e4))
    else:
        raise ValueError(f"Unsupported scheduler: {config.lr_scheduler}")
# ---------- end optimizer/scheduler ----------


# ---------- LR finder & plotting (unchanged aside from robust saving) ----------
def find_learning_rate(train_loader, model, criterion, optimizer, device, start_lr, end_lr, num_batches, scaler, logger, config):
    model.train()
    lrs = []
    losses = []
    best_loss = float('inf')

    logger.info(f"Starting LR Finder: {start_lr} -> {end_lr} over {num_batches} batches.")

    for pg in optimizer.param_groups:
        pg['lr'] = start_lr
    multiplier = (end_lr / start_lr) ** (1.0 / max(1, num_batches))
    use_amp = scaler is not None
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    if use_amp and device_type == 'cuda':
        if getattr(config, "use_bf16", False) and torch.cuda.is_bf16_supported():
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        autocast_ctx = torch.nullcontext()

    iterator = iter(train_loader)
    for batch_idx in tqdm(range(num_batches), desc="LR Finder", leave=True, dynamic_ncols=True, file=sys.stdout):
        try:
            images, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, target = next(iterator)

        if device.type == 'cuda' and getattr(config, "use_channels_last", True):
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss.item())
        for pg in optimizer.param_groups:
            pg['lr'] = pg['lr'] * multiplier

        if loss.item() < best_loss:
            best_loss = loss.item()

        if loss.item() > 4 * best_loss:
            logger.info("Loss diverged during LR finder — stopping early.")
            break

    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    os.makedirs(getattr(config, "log_dir", "."), exist_ok=True)
    lr_finder_path = os.path.join(getattr(config, "log_dir", "."), "lr_finder.png")
    plt.savefig(lr_finder_path)
    logger.info(f"LR Finder plot saved to {lr_finder_path}")
    plt.close()


def plot_metrics(start_epoch, total_epochs, train_losses, val_losses, train_acc1s, val_acc1s, lrs, log_dir, logger, eval_interval):
    import numpy as np
    epochs_in_history = list(range(start_epoch, total_epochs))
    t_losses = [float(x) if x is not None else np.nan for x in train_losses]
    v_losses = [float(x) if x is not None else np.nan for x in val_losses]
    t_acc = [float(x) if x is not None else np.nan for x in train_acc1s]
    v_acc = [float(x) if x is not None else np.nan for x in val_acc1s]

    os.makedirs(log_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot([e + 1 for e in epochs_in_history], t_losses, label='Train Loss')
    valid_val_epochs = [e + 1 for e, val in zip(epochs_in_history, v_losses) if not np.isnan(val)]
    valid_v_losses = [val for val in v_losses if not np.isnan(val)]
    if valid_v_losses:
        plt.plot(valid_val_epochs, valid_v_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([e + 1 for e in epochs_in_history], t_acc, label='Train Acc@1')
    valid_v_acc = [val for val in v_acc if not np.isnan(val)]
    if valid_v_acc:
        plt.plot(valid_val_epochs, valid_v_acc, label='Val Acc@1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(log_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Training metrics plot saved to {plot_path}")
    plt.close()

    if lrs:
        plt.figure()
        plt.plot([e + 1 for e in epochs_in_history], lrs, label='LR')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.title('Learning Rate over epochs')
        plt.grid(True)
        lr_plot_path = os.path.join(log_dir, "learning_rates.png")
        plt.savefig(lr_plot_path)
        logger.info(f"LR plot saved to {lr_plot_path}")
        plt.close()
# ---------- end plotting ----------


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description='Train ResNet-50 on ImageNet with S3 (robust)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--train-subset', type=int, default=None, help='Training subset size (for testing)')
    parser.add_argument('--val-subset', type=int, default=None, help='Validation subset size (for testing)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (local or s3 URI)')
    parser.add_argument('--find-lr', action='store_true', help='Run learning rate finder')
    parser.add_argument('--lr-finder-start-lr', type=float, default=1e-7, help='Start LR for LR finder')
    parser.add_argument('--lr-finder-end-lr', type=float, default=1.0, help='End LR for LR finder')
    parser.add_argument('--lr-finder-num-batches', type=int, default=100, help='Number of batches for LR finder')
    parser.add_argument('--use-fp16', action='store_true', help='Use FP16 instead of BF16 (override)')
    parser.add_argument('--save-every-n-batches', type=int, default=None, help='Save checkpoint every N batches')
    parser.add_argument('--log-interval', type=int, default=100, help='Log progress every N batches')
    parser.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket name (override config)')
    parser.add_argument('--s3-prefix-train', type=str, default=None, help='S3 prefix for training data')
    parser.add_argument('--s3-prefix-val', type=str, default=None, help='S3 prefix for validation data')
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory for S3 cache (not used if streaming)')
    parser.add_argument('--force-relist-s3', action='store_true', help='Force re-listing S3 files, ignoring cache')
    parser.add_argument('--s3-checkpoint-bucket', type=str, default=None, help='S3 bucket for checkpoints')
    parser.add_argument('--s3-checkpoint-prefix', type=str, default=None, help='S3 prefix for checkpoints')
    parser.add_argument('--resume-from-s3-latest', action='store_true', help='Automatically find and resume from latest S3 checkpoint')
    args = parser.parse_args()

    # Load config
    config = Config()

    # CLI overrides
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.train_subset:
        config.train_subset_size = args.train_subset
    if args.val_subset:
        config.val_subset_size = args.val_subset
    if args.resume:
        config.resume = True
        config.resume_path = args.resume
    if args.save_every_n_batches is not None:
        config.save_every_n_batches = args.save_every_n_batches
    if args.log_interval:
        config.log_interval = args.log_interval

    if args.s3_bucket:
        config.s3_bucket = args.s3_bucket
    if args.s3_prefix_train:
        config.s3_prefix_train = args.s3_prefix_train
    if args.s3_prefix_val:
        config.s3_prefix_val = args.s3_prefix_val
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.force_relist_s3:
        config.force_relist_s3 = True
    if args.s3_checkpoint_bucket:
        config.s3_checkpoint_bucket = args.s3_checkpoint_bucket
    if args.s3_checkpoint_prefix:
        config.s3_checkpoint_prefix = args.s3_checkpoint_prefix
    if args.resume_from_s3_latest:
        config.resume_from_s3_latest = True

    # Precision defaults
    config.mixed_precision = True
    config.use_bf16 = False
    if get_device(config).type == 'cuda':
        if torch.cuda.is_bf16_supported():
            config.use_bf16 = True
            if args.use_fp16:
                config.use_bf16 = False
        elif args.use_fp16:
            config.use_bf16 = False
        else:
            config.use_bf16 = False
    else:
        config.mixed_precision = False

    # Performance defaults
    config.pin_memory = getattr(config, "pin_memory", True)
    config.num_workers = getattr(config, "num_workers", max(4, (os.cpu_count() or 4) // 2))
    config.prefetch_factor = getattr(config, "prefetch_factor", 2)
    config.persistent_workers = getattr(config, "persistent_workers", True)
    config.use_channels_last = getattr(config, "use_channels_last", True)
    config.grad_accum_steps = getattr(config, "grad_accum_steps", 1)
    # dataloader timeout (seconds) - set in config to protect against S3 stalls
    config.dataloader_timeout = getattr(config, "dataloader_timeout", 900)

    # Device
    config.device = get_device(config)

    # Random seed
    set_seed(getattr(config, "seed", 42))

    # cudnn benchmark
    torch.backends.cudnn.benchmark = True

    # Logging
    logger = setup_logging(getattr(config, "log_dir", "./logs"), f"{getattr(config, 'checkpoint_name', 'resnet50')}_s3")
    logger.info("="*80)
    logger.info("Starting ResNet-50 ImageNet Training with S3 (robust)")
    logger.info("="*80)
    logger.info(f"\n{config}")
    logger.info(f"Train subset size: {getattr(config, 'train_subset_size', None)}")
    logger.info(f"Val subset size: {getattr(config, 'val_subset_size', None)}")
    logger.info(f"S3 Cache Directory: {getattr(config, 'cache_dir', None)}")
    logger.info(f"Force S3 Relist: {getattr(config, 'force_relist_s3', False)}")
    logger.info(f"S3 Checkpoint Bucket: {getattr(config, 's3_checkpoint_bucket', None)}")
    logger.info(f"S3 Checkpoint Prefix: {getattr(config, 's3_checkpoint_prefix', None)}")
    logger.info(f"Resume from S3 Latest: {getattr(config, 'resume_from_s3_latest', False)}")
    logger.info(f"Save Every N Batches: {getattr(config, 'save_every_n_batches', None)}")
    logger.info(f"Log Interval: {getattr(config, 'log_interval', 100)}")
    logger.info(f"Mixed Precision: {getattr(config, 'mixed_precision', False)} (BF16: {getattr(config, 'use_bf16', False)})")

    # Initialize NVML once (safe)
    if getattr(config, "device", None) and getattr(config.device, "type", None) == 'cuda':
        _ = get_gpu_utilization()

    # Build model
    logger.info("Creating ResNet-50 model...")
    model = resnet50(num_classes=getattr(config, "num_classes", 1000)).to(config.device)
    if config.device.type == 'cuda' and config.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")

    if getattr(config, "compile_model", False):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile()")
        except Exception as e:
            logger.warning(f"torch.compile failed or not available: {e}")

    # Initialize DataLoaders with retries (robust against transient S3 listing/network errors)
    train_loader, val_loader = None, None
    for attempt in range(getattr(config, "dataloader_retry_limit", 3)):
        try:
            train_loader, val_loader = get_data_loaders(config)
            logger.info("DataLoaders initialized successfully.")
            break
        except Exception as e:
            logger.error(f"DataLoader init failed (attempt {attempt+1}): {e}")
            traceback.print_exc()
            time.sleep(5)
    if train_loader is None or val_loader is None:
        logger.critical("Failed to initialize DataLoaders after retries, aborting.")
        sys.exit(1)

    # steps per epoch for OneCycleLR
    config.steps_per_epoch = len(train_loader)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Loss
    if getattr(config, "label_smoothing", 0.0) > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer & scheduler
    optimizer = get_optimizer(model, config)
    logger.info(f"Using optimizer: {getattr(config, 'optimizer', 'sgd')}")
    scheduler = get_lr_scheduler(optimizer, config)
    logger.info(f"Using scheduler: {getattr(config, 'lr_scheduler', 'step')}")

    # Mixed precision scaler
    scaler = GradScaler() if getattr(config, "mixed_precision", False) else None
    if getattr(config, "mixed_precision", False):
        logger.info(f"Using mixed precision (BF16={getattr(config, 'use_bf16', False)})")

    # LR finder option
    if getattr(args, "find_lr", False):
        logger.info("Running LR finder...")
        find_learning_rate(train_loader, model, criterion, optimizer, config.device,
                           args.lr_finder_start_lr, args.lr_finder_end_lr, args.lr_finder_num_batches,
                           scaler, logger, config)
        logger.info("LR Finder complete.")
        shutdown_nvml()
        return

    # Resume logic
    start_epoch = 0
    start_batch_idx = 0
    best_acc1 = 0.0
    best_train_acc1 = 0.0
    resume_checkpoint_path = None
    # CLI resume path overrides
    if getattr(config, "resume", False) and getattr(config, "resume_path", None):
        resume_checkpoint_path = config.resume_path
        logger.info(f"Resuming from specified checkpoint: {resume_checkpoint_path}")
    elif getattr(config, "resume_from_s3_latest", False):
        logger.info(f"Attempting to resume from latest S3 checkpoint in {getattr(config, 's3_checkpoint_path', '')}...")
        try:
            resume_checkpoint_path = get_latest_s3_checkpoint(config.s3_checkpoint_bucket, config.s3_checkpoint_prefix, logger)
            if resume_checkpoint_path:
                logger.info(f"Latest S3 checkpoint found: {resume_checkpoint_path}")
            else:
                logger.warning("No latest S3 checkpoint found.")
        except Exception as e:
            logger.warning(f"get_latest_s3_checkpoint failed: {e}")

    if resume_checkpoint_path:
        try:
            checkpoint = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, logger)
            start_epoch = checkpoint.get('epoch', 0)
            start_batch_idx = checkpoint.get('batch_idx', 0) + 1
            best_acc1 = checkpoint.get('best_acc1', 0.0)
            best_train_acc1 = checkpoint.get('best_train_acc1', 0.0)
            if start_batch_idx >= len(train_loader):
                start_epoch += 1
                start_batch_idx = 0
            logger.info(f"Resuming at epoch {start_epoch+1}, batch {start_batch_idx+1}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {resume_checkpoint_path}: {e}. Starting from scratch.")

    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    history_train_losses = []
    history_train_acc1s = []
    history_val_losses = []
    history_val_acc1s = []
    history_lrs = []

    # training loop
    for epoch in range(start_epoch, getattr(config, "epochs", 90)):
        logger.info(f"\nEpoch {epoch+1}/{getattr(config, 'epochs', 90)}")
        try:
            current_lr = optimizer.param_groups[0]['lr']
        except Exception:
            current_lr = getattr(config, "learning_rate", 0.0)
        history_lrs.append(current_lr)
        logger.info(f"Learning rate: {current_lr:.6f}")

        # If every few epochs, do an S3 health check and warn early
        if (epoch + 1) % 3 == 0:
            if not check_s3_access(config, logger):
                logger.warning("S3 health check failed — S3 may be slow or inaccessible. Monitor logs and nvidia-smi.")

        # Reset start_batch_idx for epochs after the initial (resume) epoch
        if epoch > start_epoch:
            start_batch_idx = 0

        with Timer(f"Epoch {epoch+1} training"):
            # train
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                train_loader, model, criterion, optimizer,
                epoch, config, logger, scaler,
                scheduler=scheduler,
                start_batch_idx=start_batch_idx,
                current_best_acc1=best_acc1,
                current_best_train_acc1=best_train_acc1
            )

        history_train_losses.append(train_loss)
        history_train_acc1s.append(train_acc1)

        # step scheduler if not OneCycle
        if getattr(config, "lr_scheduler", "").lower() != "onecycle":
            try:
                scheduler.step()
            except Exception:
                pass

        best_train_acc1 = max(train_acc1, best_train_acc1)

        # Validation
        if (epoch + 1) % getattr(config, "eval_interval", 1) == 0:
            with Timer(f"Epoch {epoch+1} validation"):
                acc1, acc5, val_loss = validate(val_loader, model, criterion, config, logger)
            history_val_losses.append(val_loss)
            history_val_acc1s.append(acc1)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            logger.info(
                f"Epoch [{epoch+1}] - Train Loss: {train_loss:.4f} Train Acc@1: {train_acc1:.2f}% "
                f"Val Loss: {val_loss:.4f} Val Acc@1: {acc1:.2f}% Val Acc@5: {acc5:.2f}% "
                f"Best Train Acc@1: {best_train_acc1:.2f}% Best Val Acc@1: {best_acc1:.2f}%"
            )
        else:
            history_val_losses.append(None)
            history_val_acc1s.append(None)
            is_best = False
            logger.info(f"Epoch [{epoch+1}] - Train Loss: {train_loss:.4f} Train Acc@1: {train_acc1:.2f}% Best Train Acc@1: {best_train_acc1:.2f}%")

        # Epoch-end checkpoint
        if (epoch + 1) % getattr(config, "save_every", 1) == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': len(train_loader) - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_acc1': best_acc1,
                'best_train_acc1': best_train_acc1,
                'config': config
            }
            fname = f'checkpoint_s3_epoch_{epoch}.pth'
            save_checkpoint(checkpoint, is_best, config.checkpoint_dir, fname,
                            s3_bucket=config.s3_checkpoint_bucket, s3_prefix=config.s3_checkpoint_prefix)
            logger.info(f"Saved epoch-end checkpoint for epoch {epoch+1} (local and S3).")

        # Log GPU memory usage (if CUDA)
        if getattr(config.device, "type", None) == 'cuda':
            try:
                idx = config.device.index if hasattr(config.device, "index") else 0
                tot_mem = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
                max_alloc = torch.cuda.max_memory_allocated(idx) / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats(idx)
                logger.info(f"GPU memory: total {tot_mem:.2f} GB, peak allocated {max_alloc:.2f} GB")
            except Exception:
                logger.warning("Failed to query GPU memory stats.")

    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best training accuracy: {best_train_acc1:.2f}%")
    logger.info(f"Best validation accuracy: {best_acc1:.2f}%")
    logger.info("="*80)

    logger.info("Generating training history plots...")
    try:
        plot_metrics(start_epoch, config.epochs, history_train_losses, history_val_losses,
                     history_train_acc1s, history_val_acc1s, history_lrs, config.log_dir, logger, getattr(config, "eval_interval", 1))
        logger.info("Training history plots generated and saved.")
    except Exception as e:
        logger.warning(f"Failed to generate training plots: {e}")

    # Shutdown NVML
    shutdown_nvml()


if __name__ == '__main__':
    main()
