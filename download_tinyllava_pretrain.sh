#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -c 1
#SBATCH --job-name=download_tinyllava
#SBATCH --mem=30G
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=/home/mila/s/sparsha.mishra/scratch/hyperalignment/slurm_logs/logs-%j.out
#SBATCH --error=/home/mila/s/sparsha.mishra/scratch/hyperalignment/slurm_errors/error-%j.err


dest="/home/mila/s/sparsha.mishra/scratch/tinyllava-pretrain/images.zip"
url="https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip"

ulimit -Sn $(ulimit -Hn)

wget -U "$dest" "$url"
