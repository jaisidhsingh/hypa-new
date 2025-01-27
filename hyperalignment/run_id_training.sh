#!/bin/bash
#SBATCH --job-name=ds_3-4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --output=/home/mila/s/sparsha.mishra/projects/hypa-new/ds_3-4.out
#SBATCH --error=/home/mila/s/sparsha.mishra/projects/hypa-new/ds_3-4.err

module load anaconda/3

conda activate /home/mila/s/sparsha.mishra/scratch/generate2

ulimit -Sn $(ulimit -Hn)

pyfile="/home/mila/s/sparsha.mishra/projects/hypa-new/hyperalignment/src/learn_hnet.py"

python3 $pyfile \
    --experiment-name="three-fourth_hnet_12-4_fmlp_c-32_bs-512_lr-1e-2" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
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
    --dataset-scale=0.75 \
    --hnet-decoder-type="mlp";

# python3 src/learn_hnet.py \
#     --experiment-name="half_hnet_12-4_fmlp_c-32_bs-512_lr-1e-2" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=20 \
#     --learning-rate=1e-2 \
#     --batch-size=512 \
#     --random-seed=0 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --dataset-scale=0.5 \
#     --hnet-decoder-type="mlp";

# python3 src/learn_hnet.py \
#     --experiment-name="one-fourth_hnet_12-4_fmlp_c-32_bs-512_lr-1e-2" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=20 \
#     --learning-rate=1e-2 \
#     --batch-size=512 \
#     --random-seed=0 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --dataset-scale=0.25 \
#     --hnet-decoder-type="mlp";

# python3 src/learn_hnet.py \
#     --experiment-name="hnet_12-4_fmlp_c-32_bs-512_lr-1e-2" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
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

# python3 src/learn_hnet.py \
#     --experiment-name="hnet_12-4_fmlp_c-32_bs-4096_lr-3e-2" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=10 \
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
#     --num-epochs=10 \
#     --learning-rate=5e-2 \
#     --batch-size=16384 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --chunk-dim=256 \
#     --hnet-decoder-type="mlp";