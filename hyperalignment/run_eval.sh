python3 evaluation.py \
    --exp-name="vit_small_patch16_224" \
    --seed=0 \
    --run-type="sep" \
    --epoch=1 \
    --encoder-index=0 \
    --benchmarks="imagenet1k";


python3 evaluation.py \
    --exp-name="norm_c_32" \
    --seed=0 \
    --run-type="mm" \
    --epoch=1 \
    --encoder-index=0 \
    --benchmarks="imagenet1k";
