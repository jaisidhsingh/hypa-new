python3 src/evaluation.py \
    --exp-name="vit_small_patch16_224" \
    --seed=0 \
    --run-type="sep" \
    --epoch=10 \
    --encoder-index=0 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

python3 src/evaluation.py \
    --exp-name="vit_base_patch16_224" \
    --seed=0 \
    --run-type="sep" \
    --epoch=10 \
    --encoder-index=10 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

python3 src/evaluation.py \
    --exp-name="vit_large_patch16_224" \
    --seed=0 \
    --run-type="sep" \
    --epoch=10 \
    --encoder-index=20 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

python3 src/evaluation.py \
    --exp-name="ie_30_mlp_c_32_norm" \
    --seed=0 \
    --run-type="mm" \
    --epoch=1 \
    --encoder-index=0 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

python3 src/evaluation.py \
    --exp-name="ie_30_mlp_c_32_norm" \
    --seed=0 \
    --run-type="mm" \
    --epoch=1 \
    --encoder-index=10 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

python3 src/evaluation.py \
    --exp-name="ie_30_mlp_c_32_norm" \
    --seed=0 \
    --run-type="mm" \
    --epoch=1 \
    --encoder-index=20 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

# python3 src/evaluation.py \
#     --exp-name="flexivit_small.300ep_in1k" \
#     --seed=0 \
#     --run-type="ood" \
#     --ood-results-path="ood_attempt_10k_avg.pt" \
#     --epoch=10 \
#     --encoder-index=5 \
#     --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

