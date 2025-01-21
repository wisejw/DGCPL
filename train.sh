#!/bin/bash

dt=$(date '+%Y%m%d_%H%M%S')
dataset="MOOC"  # Dataset name
mode="train"    # Training mode

# Training parameters
epochs=50  # Number of epochs
batch_size=16  # Batch size
lr=0.0001  # Learning rate
kd_loss_weight=1E-5  # Knowledge distillation loss weight
dropout_rate=0.5  # Dropout rate
T=0.5  # Temperature coefficient
seed=42  # Random seed

# Output parameters
echo "***** Training *****"
echo "Dataset: $dataset"
echo "Epochs: $epochs"
echo "Batch Size: $batch_size"
echo "Learning Rate: $lr"
echo "KD Loss Weight: $kd_loss_weight"
echo "Dropout Rate: $dropout_rate"
echo "Temperature: $T"
echo "Seed: $seed"
echo "******************************"

# Output directory and log file
save_dir_name="model_${dataset}_${dt}"
log="logs/${dataset}/${mode}_${save_dir_name}.log.txt"

# Create the directory for logs
mkdir -p logs

##### Training ######
python3 -u train.py \
    --epochs $epochs \
    --batch_size $batch_size \
    --lr $lr \
    --kd_loss_weight $kd_loss_weight \
    --dropout_rate $dropout_rate \
    --T $T \
    --seed $seed \
    --dataset $dataset \
    > ${log} 2>&1 &

echo "Log: ${log}"
