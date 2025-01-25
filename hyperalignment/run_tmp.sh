# python3 src/evaluation.py \
#     --exp-name="hnet_12-4_mlp_c-32_bs-512_lr-1e-2" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=20 \
#     --encoder-index=0 \
#     --image-embed-dim=384 \
#     --benchmarks="imagenet1k";

python3 src/evaluation.py \
    --exp-name="hnet_12-4_chkmlp_c-32_bs-4096-lr-3e-2" \
    --seed=0 \
    --run-type="mm" \
    --epoch=20 \
    --encoder-index=0 \
    --image-embed-dim=384 \
    --benchmarks="imagenet1k";

python3 src/evaluation.py \
    --exp-name="hnet_12-4_chkmlp_c-32_bs-16384-lr-5e-2" \
    --seed=0 \
    --run-type="mm" \
    --epoch=20 \
    --encoder-index=0 \
    --image-embed-dim=384 \
    --benchmarks="imagenet1k";