#!/bin/bash
#SBATCH --job-name=Ldim_7
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --output=/home/mila/s/sparsha.mishra/projects/hypa-new/Ldim_7.out
#SBATCH --error=/home/mila/s/sparsha.mishra/projects/hypa-new/Ldim_7.err

module load anaconda/3

conda activate /home/mila/s/sparsha.mishra/scratch/generate2

ulimit -Sn $(ulimit -Hn)

pyfile="/home/mila/s/sparsha.mishra/projects/hypa-new/hyperalignment/src/analysis.py"

python3 $pyfile --num-workers=8 --image-embed-dim=1024 --batch-size=128 --offset=7 --end=8;


# python3 src/learn_hnet.py \
#     --experiment-name="hnet_12-4_fmlp_c-32_bs-512_lr-1e-2_ep-100" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=100 \
#     --learning-rate=1e-2 \
#     --batch-size=512 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --chunk-dim=256 \
#     --hnet-decoder-type="mlp";

# python3 src/learn_hnet.py \
#     --experiment-name="hnet_12-4_fmlp_c-32_bs-4096_lr-3e-2" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=20 \
#     --learning-rate=3e-2 \
#     --batch-size=4096 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --chunk-dim=256 \
#     --hnet-decoder-type="mlp";

# python3 src/learn_hnet.py \
#     --experiment-name="hnet_12-4_fmlp_c-32_bs-16384_lr-5e-2" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=20 \
#     --learning-rate=5e-2 \
#     --batch-size=16384 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --chunk-dim=256 \
#     --hnet-decoder-type="mlp";