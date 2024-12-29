python3 ood_autodecoding.py \
    --image-encoder="maxvit_base_tf_224.in1k" \
    --image-embed-dim=768 \
    --learning-rate=1e-5 \
    --save-path="ood_attempt_10k_avg_visf_384.pt" \
    --batch-size=4;

python3 evaluation.py \
    --exp-name="maxvit_base_tf_224.in1k" \
    --seed=0 \
    --run-type="ood" \
    --ood-results-path="ood_attempt_10k_avg_visf_384.pt" \
    --image-embed-dim=768 \
    --epoch=1 \
    --encoder-index=5 \
    --benchmarks="cifar10,cifar100,imagenet1k,mscoco";
# write test