import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools # Added for efficient iteration skipping when resuming mid-epoch


# Import our modules from the local directory
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
    scheduler=None,
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
    
    # Skip batches if resuming mid-epoch
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
    # Loop over batches, starting from start_batch_idx
    for i, (images, target) in pbar:
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        images = images.to(config.device, non_blocking=True)
        target = target.to(config.device, non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast(enabled=config.mixed_precision):
            output = model(images)
            loss = criterion(output, target)
        
        # Compute accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # Record loss and accuracy
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.2f}%',
            'Acc@5': f'{top5.avg:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Log detailed progress at intervals
        if i % config.log_interval == 0:
            logger.info(
                f"Epoch [{epoch+1}] Batch [{i+1}/{len(train_loader)}] " # i+1 for 1-based indexing
                f"Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%"
            )
        
        # New: Save mid-epoch checkpoint every N batches if configured
        if config.save_every_n_batches is not None and (i + 1) % config.save_every_n_batches == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': i, # Save the completed batch index
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc1': current_best_acc1, # Use the best_acc1 from main loop
                'best_train_acc1': current_best_train_acc1, # Use the best_train_acc1 from main loop
                'config': config
            }
            save_checkpoint(
                checkpoint,
                is_best=False, # Mid-epoch saves are generally not marked as 'best'
                checkpoint_dir=config.checkpoint_dir,
                filename=f'checkpoint_s3_epoch_{epoch}_batch_{i+1}.pth', # Unique filename
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
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)):
            images = images.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)

            with autocast(enabled=config.mixed_precision):
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
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=getattr(config, "nesterov", False)
        )
    elif opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer}")
    return optimizer



def get_lr_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    if config.lr_scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
    elif config.lr_scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs
        )
    elif config.lr_scheduler.lower() == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.lr_milestones,
            gamma=config.lr_gamma
        )
    elif config.lr_scheduler.lower() == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.lr_scheduler}")
    return scheduler


def find_learning_rate(
    train_loader, model, criterion, optimizer, device,
    start_lr, end_lr, num_batches, scaler, logger
):
    """Perform a simple LR range test."""
    model.train()
    lrs = []
    losses = []
    best_loss = float('inf')
    lr_lambda = lambda x: (end_lr / start_lr) ** (x / num_batches)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for batch_idx, (images, target) in enumerate(tqdm(train_loader, total=num_batches, desc="LR Finder")):
        if batch_idx > num_batches:
            break
        images = images.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        with autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, target)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 4 * best_loss:
            logger.info("Loss diverged; stopping LR finder.")
            break

    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    lr_finder_path = os.path.join("lr_finder.png")
    plt.savefig(lr_finder_path)
    logger.info(f"LR Finder plot saved to {lr_finder_path}")
    plt.close()


def plot_metrics(start_epoch, total_epochs, train_losses, val_losses, train_acc1s, val_acc1s, lrs, log_dir, logger, eval_interval):
    """Plot training metrics."""
    epochs = list(range(start_epoch + 1, total_epochs + 1))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    if any(val_losses):
        plt.plot(epochs, [v for v in val_losses if v is not None], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc1s, label='Train Acc@1')
    if any(val_acc1s):
        plt.plot(epochs, [v for v in val_acc1s if v is not None], label='Val Acc@1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    plot_path = os.path.join(log_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Training metrics plot saved to {plot_path}")
    plt.close()



def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train ResNet-50 on ImageNet with S3')
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
    
    # S3 specific arguments
    parser.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket name')
    parser.add_argument('--s3-prefix-train', type=str, default=None, help='S3 prefix for training data')
    parser.add_argument('--s3-prefix-val', type=str, default=None, help='S3 prefix for validation data')
    
    # S3 Caching arguments
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory for S3 file list cache')
    parser.add_argument('--force-relist-s3', action='store_true', help='Force re-listing S3 files, ignoring cache')

    # S3 Checkpoint arguments (New)
    parser.add_argument('--s3-checkpoint-bucket', type=str, default=None, help='S3 bucket for checkpoints')
    parser.add_argument('--s3-checkpoint-prefix', type=str, default=None, help='S3 prefix for checkpoints')
    parser.add_argument('--resume-from-s3-latest', action='store_true', 
                        help='Automatically find and resume from the latest checkpoint in S3 checkpoint prefix')
    parser.add_argument('--save-every-n-batches', type=int, default=None, 
                        help='Save checkpoint every N batches within an epoch. Set to 0 or None to disable.')


    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.resume:
        config.resume = True
        config.resume_path = args.resume

    # Override S3 config with command line arguments
    if args.s3_bucket:
        config.s3_bucket = args.s3_bucket
    if args.s3_prefix_train:
        config.s3_prefix_train = args.s3_prefix_train
    if args.s3_prefix_val:
        config.s3_prefix_val = args.s3_prefix_val
    
    # Add subset sizes to config
    config.train_subset_size = args.train_subset
    config.val_subset_size = args.val_subset

    # Apply S3 Caching config from command line
    if args.cache_dir: # Only override if provided via command line
        config.cache_dir = args.cache_dir
    if args.force_relist_s3:
        config.force_relist_s3 = args.force_relist_s3
    
    # Apply S3 Checkpoint config from command line (New)
    if args.s3_checkpoint_bucket:
        config.s3_checkpoint_bucket = args.s3_checkpoint_bucket
    if args.s3_checkpoint_prefix:
        config.s3_checkpoint_prefix = args.s3_checkpoint_prefix
    if args.resume_from_s3_latest:
        config.resume_from_s3_latest = args.resume_from_s3_latest
    # New: Apply save_every_n_batches from command line
    if args.save_every_n_batches is not None:
        config.save_every_n_batches = args.save_every_n_batches
    
    # Set device
    config.device = get_device(config)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup logging
    logger = setup_logging(config.log_dir, f"{config.checkpoint_name}_s3")
    logger.info("="*80)
    logger.info("Starting ResNet-50 ImageNet Training with S3")
    logger.info("="*80)
    logger.info(f"\n{config}")
    logger.info(f"Train subset size: {config.train_subset_size}")
    logger.info(f"Val subset size: {config.val_subset_size}")
    logger.info(f"S3 Cache Directory: {config.cache_dir}")
    logger.info(f"Force S3 Relist: {config.force_relist_s3}")
    logger.info(f"S3 Checkpoint Bucket: {config.s3_checkpoint_bucket}")
    logger.info(f"S3 Checkpoint Prefix: {config.s3_checkpoint_prefix}")
    logger.info(f"Resume from S3 Latest: {config.resume_from_s3_latest}")
    logger.info(f"Save Every N Batches: {config.save_every_n_batches}")
    
    # Create model
    logger.info(f"\nCreating ResNet-50 model...")
    model = resnet50(num_classes=config.num_classes)
    model = model.to(config.device)
    
    # Count parameters
    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    # PyTorch 2.0 compile (optional)
    if config.compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Data loaders
    logger.info("Loading data from S3...")
    train_loader, val_loader = get_data_loaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Loss function
    if config.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer (Learning rate will be overridden by LR Finder if active)
    optimizer = get_optimizer(model, config)
    logger.info(f"Using optimizer: {config.optimizer}")
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
    if config.mixed_precision:
        logger.info("Using mixed precision training")
    
    # Initialize learning rate scheduler (moved before resume logic to ensure it's always defined)
    scheduler = get_lr_scheduler(optimizer, config)
    logger.info(f"Using scheduler: {config.lr_scheduler}")

    # Learning Rate Finder
    if args.find_lr:
        logger.info(f"\nRunning Learning Rate Finder from {args.lr_finder_start_lr:.7f} to {args.lr_finder_end_lr:.7f} for {args.lr_finder_num_batches} batches...")
        find_learning_rate(
            train_loader, model, criterion, optimizer, config.device,
            args.lr_finder_start_lr, args.lr_finder_end_lr, args.lr_finder_num_batches, scaler,
            logger
        )
        logger.info("Learning Rate Finder complete. Check the plot for optimal LR.")
        return # Exit after LR finding

    # Resume from checkpoint logic (Updated for S3 latest and mid-epoch resume)
    start_epoch = 0
    start_batch_idx = 0 # New: To resume within an epoch
    best_acc1 = 0.0
    best_train_acc1 = 0.0
    
    # Determine resume path based on priority:
    # 1. Explicit --resume path
    # 2. --resume-from-s3-latest
    resume_checkpoint_path = None
    if config.resume and config.resume_path:
        resume_checkpoint_path = config.resume_path
        logger.info(f"\nResuming from specified checkpoint: {resume_checkpoint_path}")
    elif config.resume_from_s3_latest:
        logger.info(f"\nAttempting to resume from latest S3 checkpoint in {config.s3_checkpoint_path}...")
        resume_checkpoint_path = get_latest_s3_checkpoint(config.s3_checkpoint_bucket, config.s3_checkpoint_prefix, logger)
        if resume_checkpoint_path:
            logger.info(f"Latest S3 checkpoint found: {resume_checkpoint_path}")
        else:
            logger.warning("No latest S3 checkpoint found. Starting training from scratch.")
    
    if resume_checkpoint_path:
        try:
            checkpoint = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, logger)
            start_epoch = checkpoint['epoch']
            start_batch_idx = checkpoint.get('batch_idx', 0) + 1 # Start from the next batch
            best_acc1 = checkpoint.get('best_acc1', 0.0)
            best_train_acc1 = checkpoint.get('best_train_acc1', 0.0)
            
            # If resuming mid-epoch, increment epoch and reset batch_idx if it's an epoch-end checkpoint
            # or if the batch_idx indicates completion of an epoch
            if start_batch_idx >= len(train_loader): # This means the epoch was completed
                start_epoch += 1
                start_batch_idx = 0
                logger.info(f"Resuming from end of epoch {start_epoch-1}. Starting next epoch: {start_epoch+1}")
            else:
                logger.info(f"Resuming training from epoch {start_epoch+1}, batch {start_batch_idx+1}.")

        except FileNotFoundError as e:
            logger.error(f"Resume failed: {e}. Starting training from scratch.")
        except RuntimeError as e:
            logger.error(f"Resume failed due to error loading checkpoint: {e}. Starting training from scratch.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during resume: {e}. Starting training from scratch.")
            
    else:
        logger.info("\nStarting training from scratch (no valid resume path specified).")

    # The scheduler is already initialized and potentially updated by load_checkpoint
    # No need to re-initialize here.

    # Metrics storage for plotting
    history_train_losses = []
    history_train_acc1s = []
    history_val_losses = []
    history_val_acc1s = []
    history_lrs = []


    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")
        history_lrs.append(current_lr) # Capture LR at the start of epoch

        # Reset start_batch_idx to 0 for subsequent epochs
        if epoch > start_epoch:
            start_batch_idx = 0 
        
        with Timer(f"Epoch {epoch+1} training"):
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                train_loader, model, criterion, optimizer,
                epoch, config, logger, scaler,
                scheduler=scheduler,
                start_batch_idx=start_batch_idx, # Pass start_batch_idx
                current_best_acc1=best_acc1, # Pass current best_acc1
                current_best_train_acc1=best_train_acc1 # Pass current best_train_acc1
            )
        
        history_train_losses.append(train_loss)
        history_train_acc1s.append(train_acc1)
        
        # Update learning rate
        scheduler.step()
        
        # Track best training accuracy
        best_train_acc1 = max(train_acc1, best_train_acc1)
        
        # Validate
        if (epoch + 1) % config.eval_interval == 0:
            with Timer(f"Epoch {epoch+1} validation"):
                acc1, acc5, val_loss = validate(val_loader, model, criterion, config, logger)
            
            history_val_losses.append(val_loss)
            history_val_acc1s.append(acc1)
            
            # Check if best model
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
            # If validation doesn't run for this epoch, append None for consistency in plots
            history_val_losses.append(None)
            history_val_acc1s.append(None)
            logger.info(
                f"Epoch [{epoch+1}] - Train Loss: {train_loss:.4f} "
                f"Train Acc@1: {train_acc1:.2f}% "
                f"Best Train Acc@1: {best_train_acc1:.2f}%"
            )
        
        # Save checkpoint at the end of the epoch (locally and to S3)
        # This is a separate save from the mid-epoch batch saves
        if (epoch + 1) % config.save_every == 0: # Only save epoch-end if save_every allows
            checkpoint = {
                'epoch': epoch,
                'batch_idx': len(train_loader) - 1, # Mark as last batch of epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc1': best_acc1,
                'best_train_acc1': best_train_acc1,
                'config': config # Save the entire config for reproducibility
            }
            # For epoch-end checkpoints, we use the epoch number in the filename
            save_checkpoint(
                checkpoint,
                is_best,
                config.checkpoint_dir,
                f'checkpoint_s3_epoch_{epoch}.pth',
                s3_bucket=config.s3_checkpoint_bucket,
                s3_prefix=config.s3_checkpoint_prefix
            )
            logger.info(f"Saved epoch-end checkpoint for epoch {epoch+1} (local and S3).")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best training accuracy: {best_train_acc1:.2f}%\n")
    logger.info(f"Best validation accuracy: {best_acc1:.2f}%\n")
    logger.info("="*80)

    # Plotting at the end
    logger.info("Generating training history plots...")
    plot_metrics(
        start_epoch, 
        config.epochs, 
        history_train_losses,
        history_val_losses,
        history_train_acc1s,
        history_val_acc1s,
        history_lrs,
        config.log_dir,
        logger,
        config.eval_interval
    )
    logger.info("Training history plots generated and saved.")


if __name__ == '__main__':
    main()