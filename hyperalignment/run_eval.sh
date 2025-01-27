# python3 src/evaluation.py \
#     --exp-name="vits_bs-256_lr-0.001" \
#     --seed=0 \
#     --run-type="sep" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --image-embed-dim=384 \
#     --benchmarks="cifar10,cifar100,imagenet1k";

python3 src/evaluation.py \
    --exp-name="allMiniLM_hnet_12-4_fmlp_c-32_bs-512_lr-1e-2" \
    --seed=0 \
    --run-type="mm" \
    --epoch=1 \
    --encoder-index=0 \
    --image-embed-dim=384 \
    --benchmarks="cifar10,cifar100,imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="flexivit_small.300ep_in1k" \
#     --seed=0 \
#     --run-type="ood" \
#     --ood-results-path="x.pt" \
#     --epoch=10 \
#     --encoder-index=5 \
#     --image-embed-dim=384 \
#     --benchmarks="cifar10,cifar100,imagenet1k";

