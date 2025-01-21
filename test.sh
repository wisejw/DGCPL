#!/bin/bash

dt=$(date '+%Y%m%d_%H%M%S')
dataset="MOOC"  # Dataset name
mode="test"     # Testing mode

# Testing parameters
batch_size=16  # Batch size for testing

# Output parameters
echo "***** Testing *****"
echo "Dataset: $dataset"
echo "Batch Size: $batch_size"
echo "******************************"

# Output directory and log file
save_dir_name="model_${dataset}_${dt}"
log="logs/${dataset}/${mode}_${save_dir_name}.log.txt"

# Create the directory for logs
mkdir -p logs

##### Testing ######
python3 -u test.py \
    --batch_size $batch_size \
    --dataset $dataset \
    > ${log} 2>&1 &

echo "Log: ${log}"
