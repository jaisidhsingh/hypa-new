# python3 src/evaluation.py \
#     --exp-name="vit_small_patch16_224_all-MiniLM-L12-v2" \
#     --seed=0 \
#     --run-type="sep" \
#     --epoch=1 \
#     --encoder-index=0 \
#     --image-embed-dim=384 \
#     --text-embed-dim=384 \
#     --text-encoder="all-MiniLM-L12-v2" \
#     --benchmarks="cifar10,cifar100,imagenet1k";

python3 src/evaluation.py \
    --exp-name="mm_adapt_test" \
    --seed=0 \
    --run-type="mm" \
    --epoch=1 \
    --encoder-index=0 \
    --image-embed-dim=384 \
    --text-embed-dim=768 \
    --text-encoder="sentence-t5-base" \
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

