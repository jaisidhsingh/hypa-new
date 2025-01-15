# python3 src/learn_hnet.py \
#     --experiment-name="ie_12_mlp_c_32_norm_ft_ep10_lr1e-3" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=10 \
#     --learning-rate=1e-3 \
#     --scheduler="off" \
#     --emb-loss=False \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp";

# python3 src/learn_hnet.py \
#     --experiment-name="ie_18_mlp_c_32_norm_ft_ep10_lr1e-3" \
#     --num-image-encoders=18 \
#     --encoder-batch-size=6 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=10 \
#     --learning-rate=1e-3 \
#     --scheduler="off" \
#     --emb-loss=False \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp";

# python3 src/learn_hnet.py \
#     --experiment-name="ie_24_mlp_c_32_norm_ft_ep10_lr1e-3" \
#     --num-image-encoders=24 \
#     --encoder-batch-size=8 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=10 \
#     --learning-rate=1e-3 \
#     --scheduler="off" \
#     --emb-loss=False \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp";

# python3 src/learn_hnet.py \
#     --experiment-name="ie_30_mlp_c_32_norm_ft_ep10_lr1e-3" \
#     --num-image-encoders=30 \
#     --encoder-batch-size=10 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=10 \
#     --learning-rate=1e-3 \
#     --scheduler="off" \
#     --emb-loss=False \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp";

python3 src/learn_hnet.py \
    --experiment-name="ie_12_mlp_c_32_norm_no_se" \
    --num-image-encoders=12 \
    --encoder-batch-size=4 \
    --hnet-cond-emb-dim=32 \
    --cond-type="features" \
    --num-epochs=1 \
    --learning-rate=1e-2 \
    --scheduler="off" \
    --warmup-steps=100 \
    --batch-size=512 \
    --scheduler="off" \
    --emb-loss=False \
    --normalize-output=True \
    --hnet-decoder-type="mlp";