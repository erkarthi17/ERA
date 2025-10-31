# train_s3.py
import torch
import torch.nn as nn
import torch.optim as optim
# Use torch.cuda.amp.autocast directly; GradScaler is imported from torch.cuda.amp
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools # Added for efficient iteration skipping when resuming mid-epoch
import pynvml # NEW: Import pynvml for GPU monitoring


# --- NEW: Helper for GPU monitoring ---
_nvml_initialized = False
def get_gpu_utilization(device_id: int = 0) -> float:
    global _nvml_initialized
    if not _nvml_initialized:
        try:
            pynvml.nvmlInit()
            _nvml_initialized = True
        except pynvml.NVMLError as error:
            print(f"Warning: Failed to initialize NVML: {error}. GPU utilization will not be displayed.")
            _nvml_initialized = False # Ensure it's false if init fails
            return -1.0 # Return -1 to indicate error
    
    if _nvml_initialized:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu # Returns GPU compute utilization in percent
        except pynvml.NVMLError as error:
            # print(f"Warning: Failed to get GPU utilization: {error}") # Suppress frequent warnings
            return -1.0 # Return -1 to indicate error
    return -1.0

def shutdown_nvml():
    global _nvml_initialized
    if _nvml_initialized:
        try:
            pynvml.nvmlShutdown()
            _nvml_initialized = False
        except pynvml.NVMLError as error:
            print(f"Warning: Failed to shutdown NVML: {error}.")
# --- END NEW HELPERS ---

# Local imports (assumes package layout)
from .config import Config
from .model import resnet50
from .data_s3 import get_data_loaders
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


def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    config,
    logger,
    scaler=None,
    scheduler=None, # Passed scheduler to train_one_epoch for OneCycleLR step
    start_batch_idx=0, # New: Parameter to start training from a specific batch
    current_best_acc1=0.0, # New: Pass current best_acc1 for mid-epoch checkpointing
    current_best_train_acc1=0.0 # New: Pass current best_train_acc1 for mid-epoch checkpointing
):
    """
    Train for one epoch.
    
    Args:
        train_loader: Training data loader
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        epoch: Current epoch number
        config: Configuration object
        logger: Logger instance
        scaler: Gradient scaler for mixed precision
        scheduler: Learning rate scheduler (needed for OneCycleLR per-step update)
        start_batch_idx: Batch index to start from within the epoch (for resuming mid-epoch)
        current_best_acc1: Current best validation accuracy (for saving checkpoints)
        current_best_train_acc1: Current best training accuracy (for saving checkpoints)
    
    Returns:
        Tuple of (average training loss, top1 accuracy, top5 accuracy)
    """
    batch_time = AverageMeter('Time', '6.3f')
    data_time = AverageMeter('Data', '6.3f')  
    losses = AverageMeter('Loss', '.4e')     
    top1 = AverageMeter('Acc@1', '6.2f') 
    top5 = AverageMeter('Acc@5', '6.2f')     
    
    # Switch to train mode
    model.train()
    
    # If resuming mid-epoch, skip previously processed batches
    if start_batch_idx > 0:
        logger.info(f"Resuming epoch {epoch+1} from batch {start_batch_idx+1}/{len(train_loader)}...")
        # Use itertools.islice to skip previously processed batches
        train_loader_iter = itertools.islice(train_loader, start_batch_idx, None)
    else:
        train_loader_iter = train_loader
    
    # Create tqdm progress bar
    pbar = tqdm(
        enumerate(train_loader_iter, start=start_batch_idx), # Start enumerate from start_batch_idx
        total=len(train_loader),
        initial=start_batch_idx,
        desc=f"Epoch {epoch+1}/{config.epochs}", 
        leave=False, dynamic_ncols=True
    )
    
    end = time.time()
    
    # Choose autocast context (for mixed precision)
    use_amp = getattr(config, "mixed_precision", False)
    use_bf16 = getattr(config, "use_bf16", False)
    device_type = 'cuda' if config.device.type == 'cuda' else 'cpu'

    if use_amp and device_type == 'cuda':
        if use_bf16:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) # Default to FP16 if not BF16
    else:
        autocast_ctx = torch.nullcontext()

    grad_accum_steps = getattr(config, "grad_accum_steps", 1)

    # Loop over batches, starting from start_batch_idx
    for i, (images, target) in pbar:
        current_data_time = time.time() - end
        data_time.update(current_data_time)
        
        # Move data to device and convert memory format for channels_last if requested
        if config.device.type == 'cuda' and getattr(config, "use_channels_last", True):
            images = images.to(config.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        else:
            images = images.to(config.device, non_blocking=True)
        target = target.to(config.device, non_blocking=True)
        
        with autocast_ctx:
            output = model(images)
            loss = criterion(output, target) / grad_accum_steps # Scale loss for gradient accumulation
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # Scale losses back for logging
        losses.update(loss.item() * grad_accum_steps, images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # Backward pass (with optional gradient accumulation)
        # Only zero gradients at the start of an accumulation step if accumulating
        if (i % grad_accum_steps) == 0:
            optimizer.zero_grad() # Only zero if starting a new accumulation cycle

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

            # Optionally update scheduler per step if it's OneCycleLR
            if scheduler is not None and getattr(config, "lr_scheduler", "").lower() == "onecycle":
                scheduler.step()
        
        # Measure elapsed time for the entire batch
        current_batch_time = time.time() - end
        batch_time.update(current_batch_time)
        end = time.time()
        
        # NEW: Get GPU Utilization
        gpu_util = get_gpu_utilization(config.device.index if config.device.type == 'cuda' else 0)
        gpu_util_str = f'{gpu_util:.0f}%' if gpu_util >= 0 else 'N/A'

        # Update progress bar with current metrics and new performance indicators
        # Try-except for optimizer.param_groups[0]['lr'] to avoid crash if optimizer is None
        try:
            current_lr = optimizer.param_groups[0]['lr']
        except AttributeError:
            current_lr = 0.0 # Or some other default/indicator

        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.2f}%',
            'Acc@5': f'{top5.avg:.2f}%',
            'LR': f'{current_lr:.6f}',
            'DataT': f'{data_time.val:.3f}s', # New: Data loading time for current batch
            'BatchT': f'{batch_time.val:.3f}s', # New: Total batch processing time for current batch
            'GPU%': gpu_util_str # New: GPU Utilization
        })
        
        # Log detailed progress at intervals
        if (i + 1) % getattr(config, "log_interval", 100) == 0: # Use config.log_interval, default 100
            logger.info(
                f"Epoch [{epoch+1}] Batch [{i+1}/{len(train_loader)}] " # i+1 for 1-based indexing
                f"Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}% "
                f"DataT: {data_time.val:.3f}s BatchT: {batch_time.val:.3f}s GPU%: {gpu_util_str}"
            )
        
        # New: Save mid-epoch checkpoint every N batches if configured
        if config.save_every_n_batches is not None and config.save_every_n_batches > 0 and (i + 1) % config.save_every_n_batches == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': i, # Save the completed batch index (0-based)
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_acc1': current_best_acc1, # Use the best_acc1 from main loop
                'best_train_acc1': current_best_train_acc1, # Use the best_train_acc1 from main loop
                'config': config
            }
            save_checkpoint(
                checkpoint,
                is_best=False, # Mid-epoch saves are generally not marked as 'best'
                checkpoint_dir=config.checkpoint_dir,
                filename=f'checkpoint_s3_epoch_{epoch}_batch_{i+1}.pth', # Unique filename (1-based batch number)
                s3_bucket=config.s3_checkpoint_bucket,
                s3_prefix=config.s3_checkpoint_prefix
            )
            logger.info(f"Saved mid-epoch checkpoint for epoch {epoch+1}, batch {i+1} (local and S3).")

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, config, logger):
    """Run evaluation on validation set."""
    batch_time = AverageMeter('Time', '6.3f')
    losses = AverageMeter('Loss', '.4e')
    top1 = AverageMeter('Acc@1', '6.2f')
    top5 = AverageMeter('Acc@5', '6.2f')

    model.eval()
    end = time.time()
    # Choose autocast context depending on precision config and device
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
            # Move and convert memory format if using channels_last
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

    logger.info(
        f"Validation Results - Loss: {losses.avg:.4f}, "
        f"Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%"
    )

    return top1.avg, top5.avg, losses.avg


def get_optimizer(model, config):
    """Initialize optimizer based on configuration."""
    opt_name = config.optimizer.lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=getattr(config, "momentum", 0.9),
            weight_decay=getattr(config, "weight_decay", 1e-4),
            nesterov=getattr(config, "nesterov", False)
        )
    elif opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, "weight_decay", 1e-4)
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, "weight_decay", 1e-4)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return optimizer


def get_lr_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    sched_name = getattr(config, "lr_scheduler", "step").lower()
    if sched_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(config, "lr_step_size", 30),
            gamma=getattr(config, "lr_gamma", 0.1)
        )
    elif sched_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=getattr(config, "epochs", 90)
        )
    elif sched_name == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=getattr(config, "lr_milestones", [30, 60, 80]),
            gamma=getattr(config, "lr_gamma", 0.1)
        )
    elif sched_name == 'onecycle':
        # OneCycleLR requires steps_per_epoch to be set in config
        steps_per_epoch = getattr(config, "steps_per_epoch", None)
        if steps_per_epoch is None:
            raise ValueError("onecycle scheduler requires config.steps_per_epoch to be set")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=getattr(config, "onecycle_pct_start", 0.3),
            anneal_strategy=getattr(config, "onecycle_anneal", "cos"),
            final_div_factor=getattr(config, "onecycle_final_div_factor", 1e4)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.lr_scheduler}")
    return scheduler


def find_learning_rate(
    train_loader, model, criterion, optimizer, device,
    start_lr, end_lr, num_batches, scaler, logger
):
    """Perform a simple LR range test and save a plot."""
    model.train()
    lrs = []
    losses = []
    best_loss = float('inf')

    logger.info(f"Starting LR Finder: from {start_lr:.7f} to {end_lr:.7f} over {num_batches} batches.")

    # We will use a multiplicative schedule to ramp LR from start_lr --> end_lr
    # Reset optimizer param groups
    for pg in optimizer.param_groups:
        pg['lr'] = start_lr

    multiplier = (end_lr / start_lr) ** (1.0 / max(1, num_batches))
    use_amp = scaler is not None
    # choose autocast dtype only if on CUDA
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    if use_amp and device_type == 'cuda':
        # Assume BF16 if available and not overridden by specific FP16 arg for LR finder
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) 
    else:
        autocast_ctx = torch.nullcontext()

    iterator = iter(train_loader)
    for batch_idx in tqdm(range(num_batches), desc="LR Finder", leave=True, dynamic_ncols=True, file=sys.stdout):
        try:
            images, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, target = next(iterator)

        if device.type == 'cuda' and getattr(model, "memory_format", None) is None:
            # ensure channels_last for inputs if model expects it (or if config suggests)
            if getattr(config, "use_channels_last", False):
                images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
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

        # record and update LR
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss.item())

        # update LR multiplicatively
        for pg in optimizer.param_groups:
            pg['lr'] = pg['lr'] * multiplier

        if loss.item() < best_loss:
            best_loss = loss.item()

        # early stop if loss explodes
        if loss.item() > 4 * best_loss:
            logger.info("Loss diverged during LR finder — stopping early.")
            break

    # Plot and save
    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    os.makedirs(config.log_dir, exist_ok=True)
    lr_finder_path = os.path.join(config.log_dir, "lr_finder.png")
    plt.savefig(lr_finder_path)
    logger.info(f"LR Finder plot saved to {lr_finder_path}")
    plt.close()


def plot_metrics(start_epoch, total_epochs, train_losses, val_losses, train_acc1s, val_acc1s, lrs, log_dir, logger, eval_interval):
    """Plot training and validation metrics to PNG files."""
    epochs = list(range(start_epoch, total_epochs)) # Use range(start_epoch, total_epochs) to align with history lists
    # Convert lists to numbers or NaN for plotting consistency
    import numpy as np
    t_losses = [float(x) if x is not None else np.nan for x in train_losses]
    v_losses = [float(x) if x is not None else np.nan for x in val_losses]
    t_acc = [float(x) if x is not None else np.nan for x in train_acc1s]
    v_acc = [float(x) if x is not None else np.nan for x in val_acc1s]

    os.makedirs(log_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot([e + 1 for e in epochs], t_losses, label='Train Loss') # +1 for 1-based epoch display
    # Filter out NaNs for plotting validation
    valid_val_epochs = [e + 1 for e, val in zip(epochs, v_losses) if not np.isnan(val)]
    valid_v_losses = [val for val in v_losses if not np.isnan(val)]
    if valid_v_losses:
        plt.plot(valid_val_epochs, valid_v_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot([e + 1 for e in epochs], t_acc, label='Train Acc@1') # +1 for 1-based epoch display
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

    # LR plot
    if lrs:
        plt.figure()
        # lrs list corresponds to total epochs from start_epoch, so adjust x-axis
        plt.plot([e + 1 for e in epochs], lrs, label='LR') # +1 for 1-based epoch display
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.title('Learning Rate over epochs')
        plt.grid(True)
        lr_plot_path = os.path.join(log_dir, "learning_rates.png")
        plt.savefig(lr_plot_path)
        logger.info(f"LR plot saved to {lr_plot_path}")
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ResNet-50 on ImageNet with S3 (optimized)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--train-subset', type=int, default=None, help='Training subset size (for testing)')
    parser.add_argument('--val-subset', type=int, default=None, help='Validation subset size (for testing)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--find-lr', action='store_true', help='Run learning rate finder')
    parser.add_argument('--lr-finder-start-lr', type=float, default=1e-7, help='Start LR for LR finder')
    parser.add_argument('--lr-finder-end-lr', type=float, default=1.0, help='End LR for LR finder')
    parser.add_argument('--lr-finder-num-batches', type=int, default=100, help='Number of batches for LR finder')
    parser.add_argument('--use-fp16', action='store_true', help='Use FP16 instead of BF16 (default BF16 on A10G if available)')
    parser.add_argument('--save-every-n-batches', type=int, default=None, help='Save checkpoint every N batches within an epoch. Set to 0 or None to disable.')
    parser.add_argument('--log-interval', type=int, default=100, help='Log progress every N batches.')


    # S3 args
    parser.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket name for data')
    parser.add_argument('--s3-prefix-train', type=str, default=None, help='S3 prefix for training data')
    parser.add_argument('--s3-prefix-val', type=str, default=None, help='S3 prefix for validation data')
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory for S3 file list cache')
    parser.add_argument('--force-relist-s3', action='store_true', help='Force re-listing S3 files, ignoring cache')
    parser.add_argument('--s3-checkpoint-bucket', type=str, default=None, help='S3 bucket for checkpoints')
    parser.add_argument('--s3-checkpoint-prefix', type=str, default=None, help='S3 prefix for checkpoints')
    parser.add_argument('--resume-from-s3-latest', action='store_true', help='Automatically find and resume from latest S3 checkpoint')

    args = parser.parse_args()

    # Load config
    config = Config()

    # Override config with CLI arguments
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

    # S3 overrides
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

    # Precision selection: default to BF16 if on CUDA and not overridden
    config.mixed_precision = True # Assume mixed precision is desired
    config.use_bf16 = False
    if get_device(config).type == 'cuda':
        # Check if BF16 is natively supported by the GPU (Ampere+)
        if torch.cuda.is_bf16_supported():
            config.use_bf16 = True
            if args.use_fp16: # If FP16 explicitly requested, override BF16
                config.use_bf16 = False
        elif args.use_fp16: # If BF16 not supported, but FP16 requested
            config.use_bf16 = False
        else: # No BF16 support and no FP16 requested, default to FP16
            config.use_bf16 = False # Forces FP16 on older GPUs
    else: # If not CUDA, disable mixed precision
        config.mixed_precision = False


    # Performance defaults (can be overridden in config file)
    config.pin_memory = getattr(config, "pin_memory", True)
    # Adjust num_workers for optimal S3/CPU load vs GPU
    config.num_workers = getattr(config, "num_workers", max(4, (os.cpu_count() or 4) // 2)) 
    config.prefetch_factor = getattr(config, "prefetch_factor", 2)
    config.persistent_workers = getattr(config, "persistent_workers", True)
    config.use_channels_last = getattr(config, "use_channels_last", True)
    config.grad_accum_steps = getattr(config, "grad_accum_steps", 1)

    # Set device
    config.device = get_device(config)

    # Set seed
    set_seed(getattr(config, "seed", 42))

    # cudnn benchmark
    torch.backends.cudnn.benchmark = True # Improves performance by auto-tuning algorithms

    # Logging
    logger = setup_logging(config.log_dir, f"{config.checkpoint_name}_s3")
    logger.info("="*80)
    logger.info("Starting ResNet-50 ImageNet Training with S3 (optimized)")
    logger.info("="*80)
    logger.info(f"\n{config}") # Print full config
    logger.info(f"Train subset size: {config.train_subset_size}")
    logger.info(f"Val subset size: {config.val_subset_size}")
    logger.info(f"S3 Cache Directory: {config.cache_dir}")
    logger.info(f"Force S3 Relist: {config.force_relist_s3}")
    logger.info(f"S3 Checkpoint Bucket: {config.s3_checkpoint_bucket}")
    logger.info(f"S3 Checkpoint Prefix: {config.s3_checkpoint_prefix}")
    logger.info(f"Resume from S3 Latest: {config.resume_from_s3_latest}")
    logger.info(f"Save Every N Batches: {config.save_every_n_batches}")
    logger.info(f"Log Interval: {config.log_interval}")
    logger.info(f"Mixed Precision: {config.mixed_precision} (BF16: {config.use_bf16})")


    # --- NEW: Initialize NVML at the start of main ---
    if config.device.type == 'cuda':
        get_gpu_utilization() # Call once to initialize NVML
    # --- END NEW ---

    # Build model
    logger.info("Creating ResNet-50 model...")
    model = resnet50(num_classes=config.num_classes).to(config.device)
    # Convert to channels-last for faster conv performance on Ampere / A10G
    if config.device.type == 'cuda' and config.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")

    if getattr(config, "compile_model", False):
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed or not available: {e}")

    # Data loaders
    logger.info("Loading data from S3...")
    train_loader, val_loader = get_data_loaders(config)
    # If OneCycleLR used, set steps_per_epoch for scheduler init
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
    logger.info(f"Using optimizer: {config.optimizer}")
    scheduler = get_lr_scheduler(optimizer, config)
    logger.info(f"Using scheduler: {config.lr_scheduler}")

    # Mixed precision scaler
    # Updated GradScaler call for FutureWarning
    scaler = GradScaler() if config.mixed_precision else None
    if config.mixed_precision:
        logger.info(f"Using mixed precision (BF16={config.use_bf16})")

    # LR finder
    if args.find_lr:
        logger.info("Running LR finder...")
        find_learning_rate(
            train_loader, model, criterion, optimizer, config.device,
            args.lr_finder_start_lr, args.lr_finder_end_lr, args.lr_finder_num_batches, scaler, logger
        )
        logger.info("LR Finder complete.")
        # --- NEW: Shutdown NVML if LR Finder exits early ---
        shutdown_nvml()
        # --- END NEW ---
        return # Exit after LR finding

    # Resume logic (explicit resume path > s3 latest)
    start_epoch = 0
    start_batch_idx = 0
    best_acc1 = 0.0
    best_train_acc1 = 0.0
    resume_checkpoint_path = None
    if getattr(config, "resume", False) and getattr(config, "resume_path", None):
        resume_checkpoint_path = config.resume_path
        logger.info(f"Resuming from specified checkpoint: {resume_checkpoint_path}")
    elif getattr(config, "resume_from_s3_latest", False):
        logger.info(f"Attempting to resume from latest S3 checkpoint in {config.s3_checkpoint_path}...")
        resume_checkpoint_path = get_latest_s3_checkpoint(config.s3_checkpoint_bucket, config.s3_checkpoint_prefix, logger)
        if resume_checkpoint_path:
            logger.info(f"Latest S3 checkpoint found: {resume_checkpoint_path}")
        else:
            logger.warning("No latest S3 checkpoint found. Starting from scratch.")

    if resume_checkpoint_path:
        try:
            checkpoint = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, logger)
            start_epoch = checkpoint.get('epoch', 0)
            # if mid-epoch was saved, resume from next batch
            start_batch_idx = checkpoint.get('batch_idx', 0) + 1
            best_acc1 = checkpoint.get('best_acc1', 0.0)
            best_train_acc1 = checkpoint.get('best_train_acc1', 0.0)
            
            # If start_batch_idx is >= total batches, it means the epoch was completed
            # and we should start the next epoch.
            if start_batch_idx >= len(train_loader):
                start_epoch += 1
                start_batch_idx = 0
                logger.info(f"Resuming from end of epoch {start_epoch}. Continuing at epoch {start_epoch+1}")
            else:
                logger.info(f"Resuming training from epoch {start_epoch+1}, batch {start_batch_idx+1}.")
        except Exception as e:
            logger.error(f"Failed to resume checkpoint: {e}. Starting from scratch.")

    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    history_train_losses = []
    history_train_acc1s = []
    history_val_losses = []
    history_val_acc1s = []
    history_lrs = []

    # If resuming, populate history for previous epochs for correct plotting
    # Assuming checkpoint contains history, or you fetch it
    # For now, we'll start history from current start_epoch
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Get current LR for logging and plotting. Check if optimizer is valid.
        try:
            current_lr = optimizer.param_groups[0]['lr']
        except AttributeError: # If optimizer is None
            current_lr = config.learning_rate # Fallback to initial LR from config
            logger.warning("Optimizer not initialized. Falling back to config.learning_rate for logging.")

        logger.info(f"Learning rate: {current_lr:.6f}")
        history_lrs.append(current_lr)

        # Reset batch index for epochs after the initial resume epoch
        if epoch > start_epoch:
            start_batch_idx = 0

        with Timer(f"Epoch {epoch+1} training"):
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                train_loader, model, criterion, optimizer,
                epoch, config, logger, scaler,
                scheduler=scheduler, # Pass scheduler to train_one_epoch
                start_batch_idx=start_batch_idx,
                current_best_acc1=best_acc1,
                current_best_train_acc1=best_train_acc1
            )

        history_train_losses.append(train_loss)
        history_train_acc1s.append(train_acc1)

        # Step scheduler if it's not OneCycleLR (OneCycle stepped per batch in train_one_epoch)
        if getattr(config, "lr_scheduler", "").lower() != "onecycle":
            scheduler.step()

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
                f"Epoch [{epoch+1}] - Train Loss: {train_loss:.4f} "
                f"Train Acc@1: {train_acc1:.2f}% "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc@1: {acc1:.2f}% "
                f"Val Acc@5: {acc5:.2f}% "
                f"Best Train Acc@1: {best_train_acc1:.2f}% "
                f"Best Val Acc@1: {best_acc1:.2f}%"
            )
        else:
            history_val_losses.append(None)
            history_val_acc1s.append(None)
            is_best = False
            logger.info(
                f"Epoch [{epoch+1}] - Train Loss: {train_loss:.4f} "
                f"Train Acc@1: {train_acc1:.2f}% "
                f"Best Train Acc@1: {best_train_acc1:.2f}%"
            )

        # Epoch-end checkpoint (local + S3)
        if (epoch + 1) % getattr(config, "save_every", 1) == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': len(train_loader) - 1, # Mark as last batch of epoch (0-based index)
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_acc1': best_acc1,
                'best_train_acc1': best_train_acc1,
                'config': config # Save the entire config for reproducibility
            }
            fname = f'checkpoint_s3_epoch_{epoch}.pth' # Using epoch number directly
            save_checkpoint(
                checkpoint,
                is_best,
                config.checkpoint_dir,
                fname,
                s3_bucket=config.s3_checkpoint_bucket,
                s3_prefix=config.s3_checkpoint_prefix
            )
            logger.info(f"Saved epoch-end checkpoint for epoch {epoch+1} (local and S3).")

        # Log GPU memory usage (if CUDA)
        if config.device.type == 'cuda':
            try:
                tot_mem = torch.cuda.get_device_properties(config.device.index).total_memory / (1024**3)
                max_alloc = torch.cuda.max_memory_allocated(config.device.index) / (1024**3)
                torch.cuda.reset_peak_memory_stats(config.device.index)
                logger.info(f"GPU memory: total {tot_mem:.2f} GB, peak allocated {max_alloc:.2f} GB")
            except Exception as e:
                logger.warning(f"Failed to log GPU memory stats: {e}")


    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best training accuracy: {best_train_acc1:.2f}%\n")
    logger.info(f"Best validation accuracy: {best_acc1:.2f}%\n")
    logger.info("="*80)

    logger.info("Generating training history plots...")
    plot_metrics(
        start_epoch, # Pass the actual starting epoch for plotting
        config.epochs,
        history_train_losses,
        history_val_losses,
        history_train_acc1s,
        history_val_acc1s,
        history_lrs,
        config.log_dir,
        logger,
        getattr(config, "eval_interval", 1)
    )
    logger.info("Training history plots generated and saved.")

    # --- NEW: Shutdown NVML at the end of main ---
    shutdown_nvml()
    # --- END NEW ---

if __name__ == '__main__':
    main()