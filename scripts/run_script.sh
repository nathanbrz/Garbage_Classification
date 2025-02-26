#!/bin/bash
#SBATCH --job-name=multimodal_training
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mail-user=nathan.deoliveira1@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Load the correct CUDA version
module load cuda/11.3.0  

# Initialize Conda using your custom script
source ~/software/init-conda

# Activate PyTorch virtual environment
conda activate pytorch_gpu

# Debug: Confirm that the correct Python version is loaded
echo "Python path: $(which python)"
python --version

# Run the training script
python train_model.py
