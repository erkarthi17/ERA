"""
Main training script for ResNet-50 using Hugging Face Datasets.
This script is adapted for Hugging Face and can use subsets for testing.
"""

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

# Import our modules
from config import Config
from model import resnet50
# --- MODIFIED: Import from data_hf.py ---
from data_hf import get_data_loaders
from utils import (
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
    """
    batch_time = AverageMeter('Time', '6.3f')
    data_time = AverageMeter('Data', '6.3f')  
    losses = AverageMeter('Loss', '.4e')     
    top1 = AverageMeter('Acc@1', '6.2f') 
    top5 = AverageMeter('Acc@5', '6.2f')     
    
    model.train()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", 
                leave=False, dynamic_ncols=True)
    
    end = time.time()
    for i, (images, target) in enumerate(pbar):
        data_time.update(time.time() - end)
        
        images = images.to(config.device, non_blocking=True)
        target = target.to(config.device, non_blocking=True)
        
        with autocast(enabled=config.mixed_precision):
            output = model(images)
            loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.2f}%',
            'Acc@5': f'{top5.avg:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        if i % config.log_interval == 0:
            logger.info(
                f"Epoch [{epoch+1}] Batch [{i}/{len(train_loader)}] "
                f"Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%"
            )
    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, config, logger):
    """
    Validate the model.
    """
    losses = AverageMeter('Loss', '.4e')    
    top1 = AverageMeter('Acc@1', '6.2f')
    top5 = AverageMeter('Acc@5', '6.2f')

    model.eval()
    
    pbar = tqdm(val_loader, desc="Validation", leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for i, (images, target) in enumerate(pbar):
            images = images.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)
            
            with autocast(enabled=config.mixed_precision):
                output = model(images)
                loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
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
    """Get optimizer based on config."""
    if config.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_lr_scheduler(optimizer, config):
    """Get learning rate scheduler based on config."""
    if config.lr_scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    elif config.lr_scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")


def main():
    """Main training function."""
    # --- MODIFIED: Argument parser for Hugging Face ---
    parser = argparse.ArgumentParser(description='Train ResNet-50 on ImageNet using Hugging Face Datasets')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--train-subset', type=int, default=None, help='Number of training samples for testing')
    parser.add_argument('--val-subset', type=int, default=None, help='Number of validation samples for testing')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.epochs = args.epochs
    if args.lr: config.learning_rate = args.lr
    if args.resume:
        config.resume = True
        config.resume_path = args.resume
    
    config.device = get_device(config)
    set_seed(config.seed)
    
    logger = setup_logging(config.log_dir, config.checkpoint_name)
    logger.info("="*80)
    logger.info("Starting ResNet-50 Training with Hugging Face Datasets")
    logger.info("="*80)
    logger.info(f"\n{config}")
    
    logger.info(f"\nCreating ResNet-50 model...")
    model = resnet50(num_classes=config.num_classes).to(config.device)
    
    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    if config.compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # --- MODIFIED: Call get_data_loaders from data_hf.py ---
    logger.info("\nLoading data from Hugging Face...")
    # This now correctly calls the HF data loader. The subset arguments are ignored
    # by data_hf.py when using 'imagenette', which is the desired behavior for testing.
    train_loader, val_loader = get_data_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_subset=args.train_subset,
        val_subset=args.val_subset
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing) if config.label_smoothing > 0 else nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_lr_scheduler(optimizer, config)
    
    scaler = GradScaler() if config.mixed_precision else None
    if config.mixed_precision: logger.info("Using mixed precision training")
    
    start_epoch = 0
    best_acc1 = 0.0
    if config.resume and config.resume_path:
        logger.info(f"\nResuming from checkpoint: {config.resume_path}")
        checkpoint = load_checkpoint(config.resume_path, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint.get('best_acc1', 0.0)
    
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        with Timer(f"Epoch {epoch+1} training"):
            train_loss, train_acc1, _ = train_one_epoch(
                train_loader, model, criterion, optimizer, epoch, config, logger, scaler
            )
        
        scheduler.step()
        
        is_best = False
        if (epoch + 1) % config.eval_interval == 0:
            with Timer(f"Epoch {epoch+1} validation"):
                acc1, _, val_loss = validate(val_loader, model, criterion, config, logger)
            
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            logger.info(f"Epoch [{epoch+1}] Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc1:.2f}%, Val Acc@1: {acc1:.2f}%, Best Val Acc@1: {best_acc1:.2f}%")
        else:
            logger.info(f"Epoch [{epoch+1}] Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc1:.2f}%")
        
        if (epoch + 1) % config.save_every == 0 or is_best:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc1': best_acc1,
                'config': config
            }, is_best, config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            logger.info(f"Saved checkpoint for epoch {epoch+1}")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_acc1:.2f}%")
    logger.info("="*80)


if __name__ == '__main__':
    main()