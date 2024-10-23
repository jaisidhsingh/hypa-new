# python3 evaluation.py \
#     --exp-name="vit_small_patch16_224" \
#     --seed=0 \
#     --run-type="sep" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

# python3 evaluation.py \
#     --exp-name="ie_30_mlp_c_32" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

# python3 evaluation.py \
#     --exp-name="ie_30_mlp_c_32_norm" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

# python3 evaluation.py \
#     --exp-name="ie_30_mlp_c_32_cosine" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

# python3 evaluation.py \
#     --exp-name="ie_30_mlp_c_32_norm_cosine" \
#     --seed=0 \
#     --run-type="mm" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

python3 evaluation.py \
    --exp-name="ie_30_mlp_c_32_norm_init" \
    --seed=0 \
    --run-type="mm" \
    --epoch=1 \
    --encoder-index=0 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

