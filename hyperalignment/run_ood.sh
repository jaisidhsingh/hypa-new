python3 ood_autodecoding.py \
    --image-encoder="beit_base_patch16_224.in22k_ft_in22k_in1k" \
    --image-embed-dim=768;

python3 evaluation.py \
    --exp-name="beit_base_patch16_224.in22k_ft_in22k_in1k" \
    --seed=0 \
    --run-type="ood" \
    --ood-results-path="ood_attempt_10k_avg_beit_768.pt" \
    --epoch=10 \
    --encoder-index=15 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";
# write test