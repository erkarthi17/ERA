2025-10-17 17:32:31 - resnet50_imagenet - INFO - ================================================================================
2025-10-17 17:32:31 - resnet50_imagenet - INFO - Starting ResNet-50 Training with Hugging Face Datasets
2025-10-17 17:32:31 - resnet50_imagenet - INFO - ================================================================================
2025-10-17 17:32:31 - resnet50_imagenet - INFO - 
Config:
  model_name: resnet50
  num_classes: 1000
  data_root: ./data/imagenet
  train_dir: train
  val_dir: val
  num_workers: 8
  pin_memory: True
  epochs: 50
  batch_size: 128
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0001
  lr_scheduler: step
  lr_step_size: 30
  lr_gamma: 0.1
  warmup_epochs: 5
  warmup_lr: 0.01
  image_size: 224
  min_crop_size: 0.08
  color_jitter: 0.4
  interpolation: bilinear
  val_batch_size: 256
  val_resize_size: 256
  val_center_crop_size: 224
  optimizer: sgd
  label_smoothing: 0.0
  mixup_alpha: 0.0
  cutmix_alpha: 0.0
  log_dir: ./logs
  checkpoint_dir: ./checkpoints
  checkpoint_name: resnet50_imagenet
  save_every: 10
  log_interval: 100
  eval_interval: 1
  resume: False
  resume_path: None
  device: cuda
  mixed_precision: True
  compile_model: False
  distributed: False
  local_rank: 0
  world_size: 1
  seed: 42
  print_freq: 100
  train_path: ./data/imagenet/train
  val_path: ./data/imagenet/val
2025-10-17 17:32:31 - resnet50_imagenet - INFO - 
Creating ResNet-50 model...
2025-10-17 17:32:32 - resnet50_imagenet - INFO - Model has 25,557,032 trainable parameters
2025-10-17 17:32:32 - resnet50_imagenet - INFO - 
Loading data from Hugging Face...
2025-10-17 17:32:32 - resnet50_imagenet - INFO - Train samples: 8566
2025-10-17 17:32:32 - resnet50_imagenet - INFO - Val samples: 3925
2025-10-17 17:32:32 - resnet50_imagenet - INFO - Using mixed precision training
2025-10-17 17:32:32 - resnet50_imagenet - INFO - 
================================================================================
2025-10-17 17:32:32 - resnet50_imagenet - INFO - Starting training...
2025-10-17 17:32:32 - resnet50_imagenet - INFO - ================================================================================
2025-10-17 17:32:32 - resnet50_imagenet - INFO - 
Epoch 1/50, LR: 0.100000
2025-10-17 17:32:35 - resnet50_imagenet - INFO - Epoch [1] Batch [0/67] Loss: 6.8235 Acc@1: 0.00% Acc@5: 0.00%
2025-10-17 17:33:12 - resnet50_imagenet - INFO - Validation - Loss: 5.8479 Acc@1: 11.67% Acc@5: 52.71%
2025-10-17 17:33:12 - resnet50_imagenet - INFO - Epoch [1] Train Loss: 8.1238, Train Acc@1: 10.46%, Val Acc@1: 11.67%, Best Val Acc@1: 11.67%
2025-10-17 17:33:14 - resnet50_imagenet - INFO - Saved checkpoint for epoch 1
2025-10-17 17:33:14 - resnet50_imagenet - INFO - 
Epoch 2/50, LR: 0.100000
2025-10-17 17:33:15 - resnet50_imagenet - INFO - Epoch [2] Batch [0/67] Loss: 2.7993 Acc@1: 9.38% Acc@5: 55.47%
2025-10-17 17:33:52 - resnet50_imagenet - INFO - Validation - Loss: 2.3353 Acc@1: 13.83% Acc@5: 55.77%
2025-10-17 17:33:52 - resnet50_imagenet - INFO - Epoch [2] Train Loss: 2.4657, Train Acc@1: 13.87%, Val Acc@1: 13.83%, Best Val Acc@1: 13.83%
2025-10-17 17:33:54 - resnet50_imagenet - INFO - Saved checkpoint for epoch 2
2025-10-17 17:33:54 - resnet50_imagenet - INFO - 
Epoch 3/50, LR: 0.100000
2025-10-17 17:33:55 - resnet50_imagenet - INFO - Epoch [3] Batch [0/67] Loss: 2.2112 Acc@1: 18.75% Acc@5: 67.97%
2025-10-17 17:34:32 - resnet50_imagenet - INFO - Validation - Loss: 2.2276 Acc@1: 19.92% Acc@5: 67.06%
2025-10-17 17:34:32 - resnet50_imagenet - INFO - Epoch [3] Train Loss: 2.2877, Train Acc@1: 16.94%, Val Acc@1: 19.92%, Best Val Acc@1: 19.92%
2025-10-17 17:34:34 - resnet50_imagenet - INFO - Saved checkpoint for epoch 3
2025-10-17 17:34:34 - resnet50_imagenet - INFO - 
Epoch 4/50, LR: 0.100000
2025-10-17 17:34:36 - resnet50_imagenet - INFO - Epoch [4] Batch [0/67] Loss: 2.2840 Acc@1: 18.75% Acc@5: 62.50%
2025-10-17 17:35:12 - resnet50_imagenet - INFO - Validation - Loss: 2.1481 Acc@1: 22.68% Acc@5: 71.03%
2025-10-17 17:35:12 - resnet50_imagenet - INFO - Epoch [4] Train Loss: 2.1709, Train Acc@1: 20.14%, Val Acc@1: 22.68%, Best Val Acc@1: 22.68%
2025-10-17 17:35:15 - resnet50_imagenet - INFO - Saved checkpoint for epoch 4
2025-10-17 17:35:15 - resnet50_imagenet - INFO - 
Epoch 5/50, LR: 0.100000
2025-10-17 17:35:16 - resnet50_imagenet - INFO - Epoch [5] Batch [0/67] Loss: 2.1247 Acc@1: 16.41% Acc@5: 76.56%
2025-10-17 17:35:53 - resnet50_imagenet - INFO - Validation - Loss: 2.0916 Acc@1: 23.62% Acc@5: 73.81%
2025-10-17 17:35:53 - resnet50_imagenet - INFO - Epoch [5] Train Loss: 2.1166, Train Acc@1: 21.97%, Val Acc@1: 23.62%, Best Val Acc@1: 23.62%
2025-10-17 17:35:55 - resnet50_imagenet - INFO - Saved checkpoint for epoch 5
2025-10-17 17:35:55 - resnet50_imagenet - INFO - 
Epoch 6/50, LR: 0.100000
2025-10-17 17:35:57 - resnet50_imagenet - INFO - Epoch [6] Batch [0/67] Loss: 2.1872 Acc@1: 18.75% Acc@5: 67.19%
2025-10-17 17:36:33 - resnet50_imagenet - INFO - Validation - Loss: 2.1592 Acc@1: 23.57% Acc@5: 71.34%
2025-10-17 17:36:33 - resnet50_imagenet - INFO - Epoch [6] Train Loss: 2.0916, Train Acc@1: 24.07%, Val Acc@1: 23.57%, Best Val Acc@1: 23.62%
2025-10-17 17:36:33 - resnet50_imagenet - INFO - 
Epoch 7/50, LR: 0.100000
2025-10-17 17:36:35 - resnet50_imagenet - INFO - Epoch [7] Batch [0/67] Loss: 2.0641 Acc@1: 23.44% Acc@5: 75.00%
2025-10-17 17:37:11 - resnet50_imagenet - INFO - Validation - Loss: 2.0897 Acc@1: 23.24% Acc@5: 72.99%
2025-10-17 17:37:11 - resnet50_imagenet - INFO - Epoch [7] Train Loss: 2.0742, Train Acc@1: 24.14%, Val Acc@1: 23.24%, Best Val Acc@1: 23.62%
2025-10-17 17:37:11 - resnet50_imagenet - INFO - 
Epoch 8/50, LR: 0.100000
2025-10-17 17:37:13 - resnet50_imagenet - INFO - Epoch [8] Batch [0/67] Loss: 2.0512 Acc@1: 25.78% Acc@5: 75.78%
2025-10-17 17:37:49 - resnet50_imagenet - INFO - Validation - Loss: 2.0039 Acc@1: 29.25% Acc@5: 76.89%
2025-10-17 17:37:49 - resnet50_imagenet - INFO - Epoch [8] Train Loss: 2.0422, Train Acc@1: 26.41%, Val Acc@1: 29.25%, Best Val Acc@1: 29.25%
2025-10-17 17:37:50 - resnet50_imagenet - INFO - Saved checkpoint for epoch 8
2025-10-17 17:37:50 - resnet50_imagenet - INFO - 
Epoch 9/50, LR: 0.100000
2025-10-17 17:37:52 - resnet50_imagenet - INFO - Epoch [9] Batch [0/67] Loss: 1.9025 Acc@1: 32.03% Acc@5: 82.03%
2025-10-17 17:38:28 - resnet50_imagenet - INFO - Validation - Loss: 2.0182 Acc@1: 29.99% Acc@5: 74.55%
2025-10-17 17:38:28 - resnet50_imagenet - INFO - Epoch [9] Train Loss: 1.9943, Train Acc@1: 28.15%, Val Acc@1: 29.99%, Best Val Acc@1: 29.99%
2025-10-17 17:38:29 - resnet50_imagenet - INFO - Saved checkpoint for epoch 9
2025-10-17 17:38:29 - resnet50_imagenet - INFO - 
Epoch 10/50, LR: 0.100000
2025-10-17 17:38:31 - resnet50_imagenet - INFO - Epoch [10] Batch [0/67] Loss: 1.9782 Acc@1: 30.47% Acc@5: 79.69%
2025-10-17 17:39:07 - resnet50_imagenet - INFO - Validation - Loss: 2.1229 Acc@1: 24.13% Acc@5: 72.23%
2025-10-17 17:39:07 - resnet50_imagenet - INFO - Epoch [10] Train Loss: 1.9579, Train Acc@1: 30.47%, Val Acc@1: 24.13%, Best Val Acc@1: 29.99%
2025-10-17 17:39:07 - resnet50_imagenet - INFO - Saved checkpoint for epoch 10
2025-10-17 17:39:07 - resnet50_imagenet - INFO - 
Epoch 11/50, LR: 0.100000
2025-10-17 17:39:09 - resnet50_imagenet - INFO - Epoch [11] Batch [0/67] Loss: 1.8798 Acc@1: 36.72% Acc@5: 78.91%
2025-10-17 17:39:45 - resnet50_imagenet - INFO - Validation - Loss: 2.0175 Acc@1: 28.31% Acc@5: 75.36%
2025-10-17 17:39:45 - resnet50_imagenet - INFO - Epoch [11] Train Loss: 1.9198, Train Acc@1: 31.81%, Val Acc@1: 28.31%, Best Val Acc@1: 29.99%
2025-10-17 17:39:45 - resnet50_imagenet - INFO - 
Epoch 12/50, LR: 0.100000
2025-10-17 17:39:47 - resnet50_imagenet - INFO - Epoch [12] Batch [0/67] Loss: 1.8305 Acc@1: 37.50% Acc@5: 81.25%
2025-10-17 17:40:23 - resnet50_imagenet - INFO - Validation - Loss: 1.8922 Acc@1: 32.18% Acc@5: 81.38%
2025-10-17 17:40:23 - resnet50_imagenet - INFO - Epoch [12] Train Loss: 1.8645, Train Acc@1: 34.66%, Val Acc@1: 32.18%, Best Val Acc@1: 32.18%
2025-10-17 17:40:24 - resnet50_imagenet - INFO - Saved checkpoint for epoch 12
2025-10-17 17:40:24 - resnet50_imagenet - INFO - 
Epoch 13/50, LR: 0.100000
2025-10-17 17:40:26 - resnet50_imagenet - INFO - Epoch [13] Batch [0/67] Loss: 1.7286 Acc@1: 40.62% Acc@5: 82.81%
2025-10-17 17:41:02 - resnet50_imagenet - INFO - Validation - Loss: 1.8084 Acc@1: 36.51% Acc@5: 82.96%
2025-10-17 17:41:02 - resnet50_imagenet - INFO - Epoch [13] Train Loss: 1.7917, Train Acc@1: 38.01%, Val Acc@1: 36.51%, Best Val Acc@1: 36.51%
2025-10-17 17:41:03 - resnet50_imagenet - INFO - Saved checkpoint for epoch 13
2025-10-17 17:41:03 - resnet50_imagenet - INFO - 
Epoch 14/50, LR: 0.100000
2025-10-17 17:41:05 - resnet50_imagenet - INFO - Epoch [14] Batch [0/67] Loss: 1.7377 Acc@1: 40.62% Acc@5: 85.16%
2025-10-17 17:41:41 - resnet50_imagenet - INFO - Validation - Loss: 1.9576 Acc@1: 32.43% Acc@5: 82.29%
2025-10-17 17:41:41 - resnet50_imagenet - INFO - Epoch [14] Train Loss: 1.7071, Train Acc@1: 41.45%, Val Acc@1: 32.43%, Best Val Acc@1: 36.51%
2025-10-17 17:41:41 - resnet50_imagenet - INFO - 
Epoch 15/50, LR: 0.100000
2025-10-17 17:41:43 - resnet50_imagenet - INFO - Epoch [15] Batch [0/67] Loss: 1.6325 Acc@1: 42.19% Acc@5: 89.84%
2025-10-17 17:42:19 - resnet50_imagenet - INFO - Validation - Loss: 1.6219 Acc@1: 44.99% Acc@5: 86.98%
2025-10-17 17:42:19 - resnet50_imagenet - INFO - Epoch [15] Train Loss: 1.6104, Train Acc@1: 45.26%, Val Acc@1: 44.99%, Best Val Acc@1: 44.99%
2025-10-17 17:42:20 - resnet50_imagenet - INFO - Saved checkpoint for epoch 15
2025-10-17 17:42:20 - resnet50_imagenet - INFO - 
Epoch 16/50, LR: 0.100000
2025-10-17 17:42:22 - resnet50_imagenet - INFO - Epoch [16] Batch [0/67] Loss: 1.5257 Acc@1: 47.66% Acc@5: 92.19%
2025-10-17 17:42:58 - resnet50_imagenet - INFO - Validation - Loss: 1.5840 Acc@1: 46.88% Acc@5: 87.01%
2025-10-17 17:42:58 - resnet50_imagenet - INFO - Epoch [16] Train Loss: 1.5587, Train Acc@1: 48.17%, Val Acc@1: 46.88%, Best Val Acc@1: 46.88%
2025-10-17 17:42:59 - resnet50_imagenet - INFO - Saved checkpoint for epoch 16
2025-10-17 17:42:59 - resnet50_imagenet - INFO - 
Epoch 17/50, LR: 0.100000
2025-10-17 17:43:01 - resnet50_imagenet - INFO - Epoch [17] Batch [0/67] Loss: 1.5044 Acc@1: 47.66% Acc@5: 90.62%
2025-10-17 17:43:37 - resnet50_imagenet - INFO - Validation - Loss: 1.5962 Acc@1: 47.64% Acc@5: 86.65%
2025-10-17 17:43:37 - resnet50_imagenet - INFO - Epoch [17] Train Loss: 1.4929, Train Acc@1: 50.12%, Val Acc@1: 47.64%, Best Val Acc@1: 47.64%
2025-10-17 17:43:38 - resnet50_imagenet - INFO - Saved checkpoint for epoch 17
2025-10-17 17:43:38 - resnet50_imagenet - INFO - 
Epoch 18/50, LR: 0.100000
2025-10-17 17:43:40 - resnet50_imagenet - INFO - Epoch [18] Batch [0/67] Loss: 1.3421 Acc@1: 53.91% Acc@5: 91.41%
2025-10-17 17:44:16 - resnet50_imagenet - INFO - Validation - Loss: 1.6548 Acc@1: 45.38% Acc@5: 87.69%
2025-10-17 17:44:16 - resnet50_imagenet - INFO - Epoch [18] Train Loss: 1.4191, Train Acc@1: 51.91%, Val Acc@1: 45.38%, Best Val Acc@1: 47.64%
2025-10-17 17:44:16 - resnet50_imagenet - INFO - 
Epoch 19/50, LR: 0.100000
2025-10-17 17:44:18 - resnet50_imagenet - INFO - Epoch [19] Batch [0/67] Loss: 1.3360 Acc@1: 51.56% Acc@5: 92.97%
2025-10-17 17:44:54 - resnet50_imagenet - INFO - Validation - Loss: 1.5955 Acc@1: 47.72% Acc@5: 89.50%
2025-10-17 17:44:54 - resnet50_imagenet - INFO - Epoch [19] Train Loss: 1.3807, Train Acc@1: 53.48%, Val Acc@1: 47.72%, Best Val Acc@1: 47.72%
2025-10-17 17:44:56 - resnet50_imagenet - INFO - Saved checkpoint for epoch 19
2025-10-17 17:44:56 - resnet50_imagenet - INFO - 
Epoch 20/50, LR: 0.100000
2025-10-17 17:44:57 - resnet50_imagenet - INFO - Epoch [20] Batch [0/67] Loss: 1.1865 Acc@1: 62.50% Acc@5: 91.41%
2025-10-17 17:45:33 - resnet50_imagenet - INFO - Validation - Loss: 1.4486 Acc@1: 50.75% Acc@5: 90.04%
2025-10-17 17:45:33 - resnet50_imagenet - INFO - Epoch [20] Train Loss: 1.3099, Train Acc@1: 56.54%, Val Acc@1: 50.75%, Best Val Acc@1: 50.75%
2025-10-17 17:45:34 - resnet50_imagenet - INFO - Saved checkpoint for epoch 20
2025-10-17 17:45:34 - resnet50_imagenet - INFO - 
Epoch 21/50, LR: 0.100000
2025-10-17 17:45:36 - resnet50_imagenet - INFO - Epoch [21] Batch [0/67] Loss: 1.1294 Acc@1: 64.06% Acc@5: 94.53%
2025-10-17 17:46:13 - resnet50_imagenet - INFO - Validation - Loss: 1.5060 Acc@1: 50.90% Acc@5: 88.99%
2025-10-17 17:46:13 - resnet50_imagenet - INFO - Epoch [21] Train Loss: 1.2642, Train Acc@1: 58.44%, Val Acc@1: 50.90%, Best Val Acc@1: 50.90%
2025-10-17 17:46:14 - resnet50_imagenet - INFO - Saved checkpoint for epoch 21
2025-10-17 17:46:14 - resnet50_imagenet - INFO - 
Epoch 22/50, LR: 0.100000
2025-10-17 17:46:15 - resnet50_imagenet - INFO - Epoch [22] Batch [0/67] Loss: 1.2457 Acc@1: 54.69% Acc@5: 93.75%
2025-10-17 17:46:52 - resnet50_imagenet - INFO - Validation - Loss: 1.3990 Acc@1: 53.66% Acc@5: 91.24%
2025-10-17 17:46:52 - resnet50_imagenet - INFO - Epoch [22] Train Loss: 1.2184, Train Acc@1: 59.53%, Val Acc@1: 53.66%, Best Val Acc@1: 53.66%
2025-10-17 17:46:53 - resnet50_imagenet - INFO - Saved checkpoint for epoch 22
2025-10-17 17:46:53 - resnet50_imagenet - INFO - 
Epoch 23/50, LR: 0.100000
2025-10-17 17:46:54 - resnet50_imagenet - INFO - Epoch [23] Batch [0/67] Loss: 1.2178 Acc@1: 54.69% Acc@5: 96.09%
2025-10-17 17:47:31 - resnet50_imagenet - INFO - Validation - Loss: 1.2764 Acc@1: 58.27% Acc@5: 92.48%
2025-10-17 17:47:31 - resnet50_imagenet - INFO - Epoch [23] Train Loss: 1.1644, Train Acc@1: 61.21%, Val Acc@1: 58.27%, Best Val Acc@1: 58.27%
2025-10-17 17:47:32 - resnet50_imagenet - INFO - Saved checkpoint for epoch 23
2025-10-17 17:47:32 - resnet50_imagenet - INFO - 
Epoch 24/50, LR: 0.100000
2025-10-17 17:47:34 - resnet50_imagenet - INFO - Epoch [24] Batch [0/67] Loss: 1.0925 Acc@1: 62.50% Acc@5: 96.88%
2025-10-17 17:48:10 - resnet50_imagenet - INFO - Validation - Loss: 1.3170 Acc@1: 57.38% Acc@5: 91.92%
2025-10-17 17:48:10 - resnet50_imagenet - INFO - Epoch [24] Train Loss: 1.1171, Train Acc@1: 63.80%, Val Acc@1: 57.38%, Best Val Acc@1: 58.27%
2025-10-17 17:48:10 - resnet50_imagenet - INFO - 
Epoch 25/50, LR: 0.100000
2025-10-17 17:48:12 - resnet50_imagenet - INFO - Epoch [25] Batch [0/67] Loss: 1.1850 Acc@1: 57.81% Acc@5: 97.66%
2025-10-17 17:48:48 - resnet50_imagenet - INFO - Validation - Loss: 1.3038 Acc@1: 58.50% Acc@5: 91.69%
2025-10-17 17:48:48 - resnet50_imagenet - INFO - Epoch [25] Train Loss: 1.0628, Train Acc@1: 64.97%, Val Acc@1: 58.50%, Best Val Acc@1: 58.50%
2025-10-17 17:48:49 - resnet50_imagenet - INFO - Saved checkpoint for epoch 25
2025-10-17 17:48:49 - resnet50_imagenet - INFO - 
Epoch 26/50, LR: 0.100000
2025-10-17 17:48:51 - resnet50_imagenet - INFO - Epoch [26] Batch [0/67] Loss: 1.0087 Acc@1: 64.06% Acc@5: 92.97%
2025-10-17 17:49:27 - resnet50_imagenet - INFO - Validation - Loss: 1.4839 Acc@1: 52.61% Acc@5: 89.38%
2025-10-17 17:49:27 - resnet50_imagenet - INFO - Epoch [26] Train Loss: 1.0191, Train Acc@1: 65.96%, Val Acc@1: 52.61%, Best Val Acc@1: 58.50%
2025-10-17 17:49:27 - resnet50_imagenet - INFO - 
Epoch 27/50, LR: 0.100000
2025-10-17 17:49:29 - resnet50_imagenet - INFO - Epoch [27] Batch [0/67] Loss: 1.0083 Acc@1: 66.41% Acc@5: 94.53%
2025-10-17 17:50:05 - resnet50_imagenet - INFO - Validation - Loss: 1.2817 Acc@1: 58.88% Acc@5: 91.34%
2025-10-17 17:50:05 - resnet50_imagenet - INFO - Epoch [27] Train Loss: 1.0077, Train Acc@1: 66.73%, Val Acc@1: 58.88%, Best Val Acc@1: 58.88%
2025-10-17 17:50:06 - resnet50_imagenet - INFO - Saved checkpoint for epoch 27
2025-10-17 17:50:06 - resnet50_imagenet - INFO - 
Epoch 28/50, LR: 0.100000
2025-10-17 17:50:08 - resnet50_imagenet - INFO - Epoch [28] Batch [0/67] Loss: 1.0489 Acc@1: 68.75% Acc@5: 94.53%
2025-10-17 17:50:44 - resnet50_imagenet - INFO - Validation - Loss: 1.3697 Acc@1: 57.61% Acc@5: 92.41%
2025-10-17 17:50:44 - resnet50_imagenet - INFO - Epoch [28] Train Loss: 0.9276, Train Acc@1: 69.44%, Val Acc@1: 57.61%, Best Val Acc@1: 58.88%
2025-10-17 17:50:44 - resnet50_imagenet - INFO - 
Epoch 29/50, LR: 0.100000
2025-10-17 17:50:46 - resnet50_imagenet - INFO - Epoch [29] Batch [0/67] Loss: 0.9311 Acc@1: 68.75% Acc@5: 96.88%
2025-10-17 17:51:22 - resnet50_imagenet - INFO - Validation - Loss: 1.2220 Acc@1: 60.66% Acc@5: 92.82%
2025-10-17 17:51:22 - resnet50_imagenet - INFO - Epoch [29] Train Loss: 0.8884, Train Acc@1: 69.80%, Val Acc@1: 60.66%, Best Val Acc@1: 60.66%
2025-10-17 17:51:23 - resnet50_imagenet - INFO - Saved checkpoint for epoch 29
2025-10-17 17:51:23 - resnet50_imagenet - INFO - 
Epoch 30/50, LR: 0.100000
2025-10-17 17:51:25 - resnet50_imagenet - INFO - Epoch [30] Batch [0/67] Loss: 0.7837 Acc@1: 73.44% Acc@5: 97.66%
2025-10-17 17:52:01 - resnet50_imagenet - INFO - Validation - Loss: 1.2644 Acc@1: 60.54% Acc@5: 92.99%
2025-10-17 17:52:01 - resnet50_imagenet - INFO - Epoch [30] Train Loss: 0.8404, Train Acc@1: 72.53%, Val Acc@1: 60.54%, Best Val Acc@1: 60.66%
2025-10-17 17:52:02 - resnet50_imagenet - INFO - Saved checkpoint for epoch 30
2025-10-17 17:52:02 - resnet50_imagenet - INFO - 
Epoch 31/50, LR: 0.010000
2025-10-17 17:52:03 - resnet50_imagenet - INFO - Epoch [31] Batch [0/67] Loss: 0.7142 Acc@1: 78.12% Acc@5: 97.66%
2025-10-17 17:52:40 - resnet50_imagenet - INFO - Validation - Loss: 1.0444 Acc@1: 66.75% Acc@5: 94.65%
2025-10-17 17:52:40 - resnet50_imagenet - INFO - Epoch [31] Train Loss: 0.6541, Train Acc@1: 79.09%, Val Acc@1: 66.75%, Best Val Acc@1: 66.75%
2025-10-17 17:52:41 - resnet50_imagenet - INFO - Saved checkpoint for epoch 31
2025-10-17 17:52:41 - resnet50_imagenet - INFO - 
Epoch 32/50, LR: 0.010000
2025-10-17 17:52:42 - resnet50_imagenet - INFO - Epoch [32] Batch [0/67] Loss: 0.5584 Acc@1: 79.69% Acc@5: 99.22%
2025-10-17 17:53:19 - resnet50_imagenet - INFO - Validation - Loss: 1.0666 Acc@1: 66.73% Acc@5: 94.96%
2025-10-17 17:53:19 - resnet50_imagenet - INFO - Epoch [32] Train Loss: 0.5513, Train Acc@1: 82.05%, Val Acc@1: 66.73%, Best Val Acc@1: 66.75%
2025-10-17 17:53:19 - resnet50_imagenet - INFO - 
Epoch 33/50, LR: 0.010000
2025-10-17 17:53:20 - resnet50_imagenet - INFO - Epoch [33] Batch [0/67] Loss: 0.4497 Acc@1: 85.16% Acc@5: 100.00%
2025-10-17 17:53:57 - resnet50_imagenet - INFO - Validation - Loss: 1.0971 Acc@1: 67.39% Acc@5: 94.88%
2025-10-17 17:53:57 - resnet50_imagenet - INFO - Epoch [33] Train Loss: 0.4996, Train Acc@1: 84.45%, Val Acc@1: 67.39%, Best Val Acc@1: 67.39%
2025-10-17 17:53:58 - resnet50_imagenet - INFO - Saved checkpoint for epoch 33
2025-10-17 17:53:58 - resnet50_imagenet - INFO - 
Epoch 34/50, LR: 0.010000
2025-10-17 17:53:59 - resnet50_imagenet - INFO - Epoch [34] Batch [0/67] Loss: 0.5348 Acc@1: 84.38% Acc@5: 99.22%
2025-10-17 17:54:36 - resnet50_imagenet - INFO - Validation - Loss: 1.0985 Acc@1: 67.75% Acc@5: 94.83%
2025-10-17 17:54:36 - resnet50_imagenet - INFO - Epoch [34] Train Loss: 0.4604, Train Acc@1: 85.64%, Val Acc@1: 67.75%, Best Val Acc@1: 67.75%
2025-10-17 17:54:37 - resnet50_imagenet - INFO - Saved checkpoint for epoch 34
2025-10-17 17:54:37 - resnet50_imagenet - INFO - 
Epoch 35/50, LR: 0.010000
2025-10-17 17:54:38 - resnet50_imagenet - INFO - Epoch [35] Batch [0/67] Loss: 0.3778 Acc@1: 85.94% Acc@5: 100.00%
2025-10-17 17:55:15 - resnet50_imagenet - INFO - Validation - Loss: 1.1107 Acc@1: 67.95% Acc@5: 94.90%
2025-10-17 17:55:15 - resnet50_imagenet - INFO - Epoch [35] Train Loss: 0.4310, Train Acc@1: 86.47%, Val Acc@1: 67.95%, Best Val Acc@1: 67.95%
2025-10-17 17:55:16 - resnet50_imagenet - INFO - Saved checkpoint for epoch 35
2025-10-17 17:55:16 - resnet50_imagenet - INFO - 
Epoch 36/50, LR: 0.010000
2025-10-17 17:55:17 - resnet50_imagenet - INFO - Epoch [36] Batch [0/67] Loss: 0.3238 Acc@1: 92.97% Acc@5: 100.00%
2025-10-17 17:55:54 - resnet50_imagenet - INFO - Validation - Loss: 1.1157 Acc@1: 68.15% Acc@5: 94.96%
2025-10-17 17:55:54 - resnet50_imagenet - INFO - Epoch [36] Train Loss: 0.3914, Train Acc@1: 87.87%, Val Acc@1: 68.15%, Best Val Acc@1: 68.15%
2025-10-17 17:55:55 - resnet50_imagenet - INFO - Saved checkpoint for epoch 36
2025-10-17 17:55:55 - resnet50_imagenet - INFO - 
Epoch 37/50, LR: 0.010000
2025-10-17 17:55:57 - resnet50_imagenet - INFO - Epoch [37] Batch [0/67] Loss: 0.3809 Acc@1: 88.28% Acc@5: 99.22%
2025-10-17 17:56:33 - resnet50_imagenet - INFO - Validation - Loss: 1.1750 Acc@1: 67.29% Acc@5: 94.73%
2025-10-17 17:56:33 - resnet50_imagenet - INFO - Epoch [37] Train Loss: 0.3677, Train Acc@1: 88.80%, Val Acc@1: 67.29%, Best Val Acc@1: 68.15%
2025-10-17 17:56:33 - resnet50_imagenet - INFO - 
Epoch 38/50, LR: 0.010000
2025-10-17 17:56:35 - resnet50_imagenet - INFO - Epoch [38] Batch [0/67] Loss: 0.2526 Acc@1: 92.19% Acc@5: 100.00%
2025-10-17 17:57:11 - resnet50_imagenet - INFO - Validation - Loss: 1.2150 Acc@1: 67.57% Acc@5: 94.47%
2025-10-17 17:57:11 - resnet50_imagenet - INFO - Epoch [38] Train Loss: 0.3219, Train Acc@1: 90.56%, Val Acc@1: 67.57%, Best Val Acc@1: 68.15%
2025-10-17 17:57:11 - resnet50_imagenet - INFO - 
Epoch 39/50, LR: 0.010000
2025-10-17 17:57:13 - resnet50_imagenet - INFO - Epoch [39] Batch [0/67] Loss: 0.3118 Acc@1: 90.62% Acc@5: 100.00%
2025-10-17 17:57:49 - resnet50_imagenet - INFO - Validation - Loss: 1.1928 Acc@1: 68.43% Acc@5: 94.39%
2025-10-17 17:57:49 - resnet50_imagenet - INFO - Epoch [39] Train Loss: 0.2925, Train Acc@1: 91.30%, Val Acc@1: 68.43%, Best Val Acc@1: 68.43%
2025-10-17 17:57:50 - resnet50_imagenet - INFO - Saved checkpoint for epoch 39
2025-10-17 17:57:50 - resnet50_imagenet - INFO - 
Epoch 40/50, LR: 0.010000
2025-10-17 17:57:52 - resnet50_imagenet - INFO - Epoch [40] Batch [0/67] Loss: 0.1949 Acc@1: 95.31% Acc@5: 99.22%
2025-10-17 17:58:29 - resnet50_imagenet - INFO - Validation - Loss: 1.2229 Acc@1: 68.54% Acc@5: 94.75%
2025-10-17 17:58:29 - resnet50_imagenet - INFO - Epoch [40] Train Loss: 0.2631, Train Acc@1: 92.35%, Val Acc@1: 68.54%, Best Val Acc@1: 68.54%
2025-10-17 17:58:30 - resnet50_imagenet - INFO - Saved checkpoint for epoch 40
2025-10-17 17:58:30 - resnet50_imagenet - INFO - 
Epoch 41/50, LR: 0.010000
2025-10-17 17:58:31 - resnet50_imagenet - INFO - Epoch [41] Batch [0/67] Loss: 0.1518 Acc@1: 96.88% Acc@5: 100.00%
2025-10-17 17:59:08 - resnet50_imagenet - INFO - Validation - Loss: 1.2850 Acc@1: 68.46% Acc@5: 94.62%
2025-10-17 17:59:08 - resnet50_imagenet - INFO - Epoch [41] Train Loss: 0.2262, Train Acc@1: 93.51%, Val Acc@1: 68.46%, Best Val Acc@1: 68.54%
2025-10-17 17:59:08 - resnet50_imagenet - INFO - 
Epoch 42/50, LR: 0.010000
2025-10-17 17:59:09 - resnet50_imagenet - INFO - Epoch [42] Batch [0/67] Loss: 0.1577 Acc@1: 96.88% Acc@5: 99.22%
2025-10-17 17:59:46 - resnet50_imagenet - INFO - Validation - Loss: 1.3166 Acc@1: 68.54% Acc@5: 94.14%
2025-10-17 17:59:46 - resnet50_imagenet - INFO - Epoch [42] Train Loss: 0.1946, Train Acc@1: 94.77%, Val Acc@1: 68.54%, Best Val Acc@1: 68.54%
2025-10-17 17:59:46 - resnet50_imagenet - INFO - 
Epoch 43/50, LR: 0.010000
2025-10-17 17:59:47 - resnet50_imagenet - INFO - Epoch [43] Batch [0/67] Loss: 0.1536 Acc@1: 96.88% Acc@5: 100.00%
2025-10-17 18:00:24 - resnet50_imagenet - INFO - Validation - Loss: 1.3394 Acc@1: 68.46% Acc@5: 94.27%
2025-10-17 18:00:24 - resnet50_imagenet - INFO - Epoch [43] Train Loss: 0.1735, Train Acc@1: 95.41%, Val Acc@1: 68.46%, Best Val Acc@1: 68.54%
2025-10-17 18:00:24 - resnet50_imagenet - INFO - 
Epoch 44/50, LR: 0.010000
2025-10-17 18:00:26 - resnet50_imagenet - INFO - Epoch [44] Batch [0/67] Loss: 0.0910 Acc@1: 99.22% Acc@5: 100.00%
2025-10-17 18:01:02 - resnet50_imagenet - INFO - Validation - Loss: 1.3789 Acc@1: 68.18% Acc@5: 94.11%
2025-10-17 18:01:02 - resnet50_imagenet - INFO - Epoch [44] Train Loss: 0.1510, Train Acc@1: 96.22%, Val Acc@1: 68.18%, Best Val Acc@1: 68.54%
2025-10-17 18:01:02 - resnet50_imagenet - INFO - 
Epoch 45/50, LR: 0.010000
2025-10-17 18:01:04 - resnet50_imagenet - INFO - Epoch [45] Batch [0/67] Loss: 0.1091 Acc@1: 98.44% Acc@5: 100.00%
2025-10-17 18:01:40 - resnet50_imagenet - INFO - Validation - Loss: 1.3782 Acc@1: 68.48% Acc@5: 94.01%
2025-10-17 18:01:40 - resnet50_imagenet - INFO - Epoch [45] Train Loss: 0.1131, Train Acc@1: 97.67%, Val Acc@1: 68.48%, Best Val Acc@1: 68.54%
2025-10-17 18:01:40 - resnet50_imagenet - INFO - 
Epoch 46/50, LR: 0.010000
2025-10-17 18:01:42 - resnet50_imagenet - INFO - Epoch [46] Batch [0/67] Loss: 0.0968 Acc@1: 98.44% Acc@5: 100.00%
2025-10-17 18:02:18 - resnet50_imagenet - INFO - Validation - Loss: 1.4297 Acc@1: 67.75% Acc@5: 94.17%
2025-10-17 18:02:18 - resnet50_imagenet - INFO - Epoch [46] Train Loss: 0.0998, Train Acc@1: 97.84%, Val Acc@1: 67.75%, Best Val Acc@1: 68.54%
2025-10-17 18:02:18 - resnet50_imagenet - INFO - 
Epoch 47/50, LR: 0.010000
2025-10-17 18:02:20 - resnet50_imagenet - INFO - Epoch [47] Batch [0/67] Loss: 0.0959 Acc@1: 96.88% Acc@5: 100.00%
2025-10-17 18:02:56 - resnet50_imagenet - INFO - Validation - Loss: 1.4375 Acc@1: 68.89% Acc@5: 94.17%
2025-10-17 18:02:56 - resnet50_imagenet - INFO - Epoch [47] Train Loss: 0.0964, Train Acc@1: 97.65%, Val Acc@1: 68.89%, Best Val Acc@1: 68.89%
2025-10-17 18:02:57 - resnet50_imagenet - INFO - Saved checkpoint for epoch 47
2025-10-17 18:02:57 - resnet50_imagenet - INFO - 
Epoch 48/50, LR: 0.010000
2025-10-17 18:02:59 - resnet50_imagenet - INFO - Epoch [48] Batch [0/67] Loss: 0.1350 Acc@1: 96.09% Acc@5: 99.22%
2025-10-17 18:03:36 - resnet50_imagenet - INFO - Validation - Loss: 1.4579 Acc@1: 68.31% Acc@5: 93.91%
2025-10-17 18:03:36 - resnet50_imagenet - INFO - Epoch [48] Train Loss: 0.0709, Train Acc@1: 98.77%, Val Acc@1: 68.31%, Best Val Acc@1: 68.89%
2025-10-17 18:03:36 - resnet50_imagenet - INFO - 
Epoch 49/50, LR: 0.010000
2025-10-17 18:03:37 - resnet50_imagenet - INFO - Epoch [49] Batch [0/67] Loss: 0.0577 Acc@1: 98.44% Acc@5: 100.00%
2025-10-17 18:04:14 - resnet50_imagenet - INFO - Validation - Loss: 1.5286 Acc@1: 68.28% Acc@5: 94.22%
2025-10-17 18:04:14 - resnet50_imagenet - INFO - Epoch [49] Train Loss: 0.0613, Train Acc@1: 98.97%, Val Acc@1: 68.28%, Best Val Acc@1: 68.89%
2025-10-17 18:04:14 - resnet50_imagenet - INFO - 
Epoch 50/50, LR: 0.010000
2025-10-17 18:04:15 - resnet50_imagenet - INFO - Epoch [50] Batch [0/67] Loss: 0.0444 Acc@1: 99.22% Acc@5: 100.00%
2025-10-17 18:04:52 - resnet50_imagenet - INFO - Validation - Loss: 1.5306 Acc@1: 68.41% Acc@5: 93.73%
2025-10-17 18:04:52 - resnet50_imagenet - INFO - Epoch [50] Train Loss: 0.0491, Train Acc@1: 99.29%, Val Acc@1: 68.41%, Best Val Acc@1: 68.89%
2025-10-17 18:04:52 - resnet50_imagenet - INFO - Saved checkpoint for epoch 50
2025-10-17 18:04:52 - resnet50_imagenet - INFO - 
================================================================================
2025-10-17 18:04:52 - resnet50_imagenet - INFO - Training completed!
2025-10-17 18:04:52 - resnet50_imagenet - INFO - Best validation accuracy: 68.89%
2025-10-17 18:04:52 - resnet50_imagenet - INFO - ================================================================================