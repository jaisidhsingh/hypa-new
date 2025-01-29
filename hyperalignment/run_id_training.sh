#!/bin/bash
#SBATCH --job-name=30_10
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --output=/home/mila/s/sparsha.mishra/projects/hypa-new/30_10.out
#SBATCH --error=/home/mila/s/sparsha.mishra/projects/hypa-new/30_10.err

# module load anaconda/3

# conda activate /home/mila/s/sparsha.mishra/scratch/generate2

ulimit -Sn $(ulimit -Hn)

pyfile="/home/mila/s/sparsha.mishra/projects/hypa-new/hyperalignment/src/learn_hnet.py"


python3 $pyfile \
    --experiment-name="lessb2_hnet_18-6_fmlp_c-32_bs-512_lr-1e-2" \
    --num-image-encoders=18 \
    --encoder-batch-size=6 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=20 \
    --learning-rate=1e-2 \
    --batch-size=512 \
    --random-seed=0 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --dataset-scale=1.0 \
    --beta2=0.95 \
    --hnet-decoder-type="mlp";

# python3 $pyfile \
#     --experiment-name="hnet_24-8_fmlp_c-32_bs-512_lr-1e-2" \
#     --num-image-encoders=24 \
#     --encoder-batch-size=8 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=20 \
#     --learning-rate=1e-2 \
#     --batch-size=512 \
#     --random-seed=2 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --chunk-dim=256 \
#     --hnet-decoder-type="mlp";

# python3 $pyfile \
#     --experiment-name="hnet_30-10_fmlp_c-32_bs-512_lr-1e-2" \
#     --num-image-encoders=30 \
#     --encoder-batch-size=10 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=20 \
#     --learning-rate=1e-2 \
#     --batch-size=512 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --chunk-dim=256 \
#     --hnet-decoder-type="mlp";
 
