python3 src/learn_hnet.py \
    --experiment-name="hnet_12-4_fmlp_c-32_bs-1024_lr-2e-2" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=10 \
    --learning-rate=2e-2 \
    --batch-size=1024 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --chunk-dim=256 \
    --hnet-decoder-type="mlp";

python3 src/learn_hnet.py \
    --experiment-name="hnet_12-4_fmlp_c-32_bs-4096_lr-3e-2" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=10 \
    --learning-rate=3e-2 \
    --batch-size=4096 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --chunk-dim=256 \
    --hnet-decoder-type="mlp";
 
python3 src/learn_hnet.py \
    --experiment-name="hnet_12-4_fmlp_c-32_bs-16384_lr-5e-2" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=10 \
    --learning-rate=5e-2 \
    --batch-size=16384 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --chunk-dim=256 \
    --hnet-decoder-type="mlp";