python3 ood_autodecoding.py \
    --image-encoder="visformer_tiny.in1k" \
    --image-embed-dim=384 \
    --learning-rate=1e-4 \
    --save-path="ood_attempt_10k_avg_visf_384.pt" \
    --batch-size=8;

python3 evaluation.py \
    --exp-name="visformer_tiny.in1k" \
    --seed=0 \
    --run-type="ood" \
    --ood-results-path="ood_attempt_10k_avg_visf_384.pt" \
    --image-embed-dim=384 \
    --epoch=1 \
    --encoder-index=6 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";
# write test