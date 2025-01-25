python3 src/learn_hnet.py \
    --experiment-name="hnet_12-4_chkmlp_c-32_bs-256_lr-3e-3" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=20 \
    --learning-rate=3e-3 \
    --batch-size=256 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --chunk-dim=256 \
    --hnet-decoder-type="chunked_mlp";

python3 src/learn_hnet.py \
    --experiment-name="hnet_12-4_mlp_c-32_bs-512_lr-1e-2" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=20 \
    --learning-rate=1e-2 \
    --batch-size=512 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --chunk-dim=256 \
    --hnet-decoder-type="chunked_mlp";
    
python3 src/learn_hnet.py \
    --experiment-name="hnet_12-4_chkmlp_c-32_bs-1024_lr-2e-2" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=20 \
    --learning-rate=2e-2 \
    --batch-size=1024 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --chunk-dim=256 \
    --hnet-decoder-type="chunked_mlp";

python3 src/learn_hnet.py \
    --experiment-name="hnet_12-4_chkmlp_c-32_bs-4096-lr-3e-2" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=20 \
    --learning-rate=3e-2 \
    --batch-size=4096 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --chunk-dim=256 \
    --hnet-decoder-type="chunked_mlp";

python3 src/learn_hnet.py \
    --experiment-name="hnet_12-4_chkmlp_c-32_bs-16384-lr-5e-2" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=20 \
    --learning-rate=5e-2 \
    --batch-size=16384 \
    --scheduler="off" \
    --warmup-steps=100 \
    --emb-loss=False \
    --normalize-output=True \
    --chunk-dim=256 \
    --hnet-decoder-type="chunked_mlp";