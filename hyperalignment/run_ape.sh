python3 src/learn_separate_mapping.py \
    --experiment-name="vit_small_patch16_224" \
    --image-encoder="vit_small_patch16_224" \
    --image-embed-dim=384 \
    --num-epochs=20;


python3 src/learn_separate_mapping.py \
    --experiment-name="vit_base_patch16_224" \
    --image-encoder="vit_base_patch16_224" \
    --image-embed-dim=768 \
    --num-epochs=20;


python3 src/learn_separate_mapping.py \
    --experiment-name="vit_large_patch16_224" \
    --image-encoder="vit_large_patch16_224" \
    --image-embed-dim=1024 \
    --num-epochs=20;
