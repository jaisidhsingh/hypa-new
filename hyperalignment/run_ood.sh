python3 src/ood_autodecoding.py \
    --image-encoder="visformer_tiny.in1k" \
    --image-embed-dim=384 \
    --hnet-ckpt-name="ie_12_mlp_c_32_norm" \
    --hnet-ckpt-num-ie=12 \
    --learning-rate=1e-4 \
    --save-path="x.pt" \
    --batch-size=8;

# python3 evaluation.py \
#     --exp-name="efficientvit_m5.r224_in1k" \
#     --seed=0 \
#     --run-type="ood" \
#     --ood-results-path="x.pt" \
#     --image-embed-dim=768 \
#     --epoch=1 \
#     --encoder-index=6 \
#     --benchmarks="cifar10,cifar100,mscoco,imagenet1k";
