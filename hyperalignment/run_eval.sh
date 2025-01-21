# python3 src/evaluation.py \
#     --exp-name="flexivit_small.300ep_in1k" \
#     --seed=0 \
#     --run-type="sep" \
#     --epoch=1 \
#     --encoder-index=5 \
#     --image-embed-dim=384 \
#     --benchmarks="cifar10,cifar100,imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="vit_small_patch16_224" \
#     --seed=0 \
#     --run-type="sep" \
#     --epoch=1 \
#     --image-embed-dim=384 \
#     --encoder-index=0 \
#     --benchmarks="imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="vit_small_patch16_224" \
#     --seed=0 \
#     --run-type="sep" \
#     --epoch=1 \
#     --image-embed-dim=1024 \
#     --encoder-index=0 \
#     --benchmarks="imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="ie_12_mlp_c_32_norm_ft_ep10_lr1e-3" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=5 \
#     --encoder-index=0 \
#     --image-embed-dim=384 \
#     --benchmarks="cifar10,cifar100,imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="ie_12_mlp_c_32_norm_ft_ep10_lr1e-3" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=10 \
#     --encoder-index=0 \
#     --image-embed-dim=384 \
#     --benchmarks="cifar10,cifar100,imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="ie_30_mlp_c_32_norm" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --image-embed-dim=768 \
#     --benchmarks="cifar10,cifar100,imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="ie_12_mlp_c_32_norm_chunked_128" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --image-embed-dim=384 \
#     --benchmarks="imagenet1k";

python3 src/evaluation.py \
    --exp-name="ie_12_mlp_c_32_norm_chunked_256_inproj" \
    --seed=0 \
    --run-type="mm" \
    --epoch=1 \
    --encoder-index=0 \
    --image-embed-dim=384 \
    --benchmarks="imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="ie_12_mlp_c_32_norm" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --image-embed-dim=384 \
#     --benchmarks="imagenet1k";

# python3 src/evaluation.py \
#     --exp-name="flexivit_small.300ep_in1k" \
#     --seed=0 \
#     --run-type="ood" \
#     --ood-results-path="x.pt" \
#     --epoch=10 \
#     --encoder-index=5 \
#     --image-embed-dim=384 \
#     --benchmarks="cifar10,cifar100,imagenet1k";

