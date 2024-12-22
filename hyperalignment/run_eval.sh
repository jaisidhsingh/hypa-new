# python3 evaluation.py \
#     --exp-name="vit_small_patch16_224" \
#     --seed=0 \
#     --run-type="sep" \
#     --epoch=10 \
#     --encoder-index=0 \
#     --benchmarks="cifar10,cifar100,imagenet1k,mscoco";

python3 evaluation.py \
    --exp-name="flexivit_small.300ep_in1k" \
    --seed=0 \
    --run-type="ood" \
    --epoch=10 \
    --encoder-index=5 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";
