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
import matplotlib.pyplot as plt # Corrected typo from matlib.pyplot

# Import our modules from the local directory
from .config import Config
from .model import resnet50 # Assuming model.py is in parent directory
from .data_s3 import get_data_loaders  # Use S3 data loader
from .utils import ( # Assuming utils.py is in parent directory
    setup_logging,
    AverageMeter,
    ProgressMeter,
    accuracy,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_device,
    count_parameters,
    Timer
)


def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    config,
    logger,
    scaler=None
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
    
    # Create tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", 
                leave=False, dynamic_ncols=True)
    
    end = time.time()
    for i, (images, target) in enumerate(pbar):
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
                f"Epoch [{epoch+1}] Batch [{i}/{len(train_loader)}] "
                f"Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%"
            )
    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, config, logger):
    """
    Validate the model.
    
    Args:
        val_loader: Validation data loader
        model: Model to validate
        criterion: Loss function
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Tuple of (top-1 accuracy, top-5 accuracy, average loss)
    """
    batch_time = AverageMeter('Time', '6.3f') 
    losses = AverageMeter('Loss', '.4e')    
    top1 = AverageMeter('Acc@1', '6.2f')
    top5 = AverageMeter('Acc@5', '6.2f')

    # Switch to evaluation mode
    model.eval()
    
    # Create tqdm progress bar for validation
    pbar = tqdm(val_loader, desc="Validation", leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(pbar):
            # Move data to device
            images = images.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)
            
            # Compute output
            with autocast(enabled=config.mixed_precision):
                output = model(images)
                loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}%',
                'Acc@5': f'{top5.avg:.2f}%'
            })
        
        logger.info(
            f"Validation - Loss: {losses.avg:.4f} "
            f"Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%"
        )
    
    return top1.avg, top5.avg, losses.avg


def get_optimizer(model, config):
    """
    Get optimizer based on config.
    
    Args:
        model: Model to optimize
        config: Configuration object
    
    Returns:
        Optimizer instance
    """
    if config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def get_lr_scheduler(optimizer, config):
    """
    Get learning rate scheduler based on config.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration object
    
    Returns:
        Learning rate scheduler
    """
    if config.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
    elif config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")
    
    return scheduler

def find_learning_rate(
    train_loader, model, criterion, optimizer, device,
    start_lr, end_lr, num_batches, scaler
):
    """
    Runs a learning rate finder.
    """
    model.train()
    lrs = []
    losses = []
    
    # Temporarily set the learning rate to start_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr

    # Exponentially increase learning rate
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(end_lr/start_lr)**(1/(num_batches-1)))

    for i, (images, target) in enumerate(train_loader):
        if i >= num_batches:
            break

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward pass with mixed precision
        with autocast(enabled=(scaler is not None)):
            output = model(images)
            loss = criterion(output, target)
        
        # Record LR and loss
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Step LR
        lr_scheduler.step()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lr_finder_plot.png'))
    plt.show() # Display the plot (might not show on EC2 without X forwarding)
    plt.close() # Close figure to free memory


def plot_metrics(start_epoch, total_epochs, train_losses, val_losses, train_acc1s, val_acc1s, lrs, log_dir, logger, eval_interval):
    """
    Generates and saves plots for training and validation metrics.
    """
    epochs_for_plot = list(range(start_epoch + 1, total_epochs + 1))

    # Filter validation metrics for plotting if eval_interval > 1 or some epochs had no validation
    val_epochs_to_plot = []
    filtered_val_losses = []
    filtered_val_acc1s = []

    for i, current_epoch_num in enumerate(epochs_for_plot):
        # Check if validation was performed for this specific epoch in the current run
        # and if the stored value is not None (in case of eval_interval > 1)
        if val_losses[i] is not None: 
            val_epochs_to_plot.append(current_epoch_num)
            filtered_val_losses.append(val_losses[i])
            filtered_val_acc1s.append(val_acc1s[i])

    # Plot Loss
    plt.figure(figsize=(14, 6)) # Wider figure to accommodate two subplots easily
    plt.subplot(1, 2, 1)
    plt.plot(epochs_for_plot, train_losses, label='Training Loss')
    if filtered_val_losses: # Only plot if there's actual validation data
        plt.plot(val_epochs_to_plot, filtered_val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_for_plot, train_acc1s, label='Training Acc@1')
    if filtered_val_acc1s: # Only plot if there's actual validation data
        plt.plot(val_epochs_to_plot, filtered_val_acc1s, label='Validation Acc@1')
    plt.title('Training and Validation Accuracy (Top-1)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plot_path_loss_acc = os.path.join(log_dir, 'training_history_loss_acc.png')
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.savefig(plot_path_loss_acc)
    logger.info(f"Saved training history (loss/accuracy) plot to {plot_path_loss_acc}")
    plt.close() # Close figure to free memory

    # Plot Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_for_plot, lrs, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    lr_plot_path = os.path.join(log_dir, 'learning_rate_schedule.png')
    plt.savefig(lr_plot_path)
    logger.info(f"Saved learning rate schedule plot to {lr_plot_path}")
    plt.close() # Close figure


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
    parser.add_argument('--find-lr', action='store_true', help='Run learning rate finder') # New argument for LR Finder
    parser.add_argument('--lr-finder-start-lr', type=float, default=1e-7, help='Start LR for LR finder') # New argument for LR Finder
    parser.add_argument('--lr-finder-end-lr', type=float, default=1.0, help='End LR for LR finder') # New argument for LR Finder
    parser.add_argument('--lr-finder-num-batches', type=int, default=100, help='Number of batches for LR finder') # New argument for LR Finder
    
    # S3 specific arguments
    parser.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket name')
    parser.add_argument('--s3-prefix-train', type=str, default=None, help='S3 prefix for training data')
    parser.add_argument('--s3-prefix-val', type=str, default=None, help='S3 prefix for validation data')
    
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
    scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None # Updated GradScaler call for FutureWarning
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
            args.lr_finder_start_lr, args.lr_finder_end_lr, args.lr_finder_num_batches, scaler
        )
        logger.info("Learning Rate Finder complete. Check the plot for optimal LR.")
        return # Exit after LR finding

    # Resume from checkpoint
    start_epoch = 0
    best_acc1 = 0.0
    best_train_acc1 = 0.0
    if config.resume and config.resume_path:
        logger.info(f"\nResuming from checkpoint: {config.resume_path}")
        # Pass the already initialized scheduler
        checkpoint = load_checkpoint(config.resume_path, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint.get('best_acc1', 0.0)
        best_train_acc1 = checkpoint.get('best_train_acc1', 0.0)
    
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

        with Timer(f"Epoch {epoch+1} training"):
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                train_loader, model, criterion, optimizer,
                epoch, config, logger, scaler
            )
        
        history_train_losses.append(train_loss)
        history_train_acc1s.append(train_acc1)
        
        # Update learning rate
        scheduler.step()
        
        # Track best training accuracy
        best_train_acc1 = max(train_acc1, best_train_acc1)
        
        # Validate
        if (epoch + 1) % config.eval_interval == 0:
            with Timer(f"Epoch {epoch+1} validation"):\
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
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc1': best_acc1,
                'best_train_acc1': best_train_acc1,
                'config': config
            }
            save_checkpoint(
                checkpoint,
                is_best,
                config.checkpoint_dir,
                f'checkpoint_s3_epoch_{epoch}.pth'
            )
            logger.info(f"Saved checkpoint for epoch {epoch+1}")
    
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