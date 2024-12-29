python3 ood_autodecoding.py \
    --image-encoder="deit3_base_patch16_224.fb_in22k_ft_in1k" \
    --image-embed-dim=768 \
    --learning-rate=1e-4 \
    --save-path="ood_attempt_10k_avg_visf_384.pt" \
    --batch-size=8;

python3 evaluation.py \
    --exp-name="deit3_base_patch16.fb_in22k_ft_in1k" \
    --seed=0 \
    --run-type="ood" \
    --ood-results-path="ood_attempt_10k_avg_visf_384.pt" \
    --image-embed-dim=768 \
    --epoch=1 \
    --encoder-index=5 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";
# write test