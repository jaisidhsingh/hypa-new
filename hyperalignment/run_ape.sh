python3 src/train_ape.py \
    --experiment-name="vit_small_patch16_224_all-MiniLM-L12-v2" \
    --image-encoder="vit_small_patch16_224" \
    --image-embed-dim=384 \
    --num-epochs=1 \
    --saving=False \
    --text-encoder="all-MiniLM-L12-v2" \
    --text-embed-dim=384;

python3 src/train_ape.py \
    --experiment-name="vit_small_patch16_224_sentence-t5-base" \
    --image-encoder="vit_small_patch16_224" \
    --image-embed-dim=384 \
    --num-epochs=1 \
    --saving=False \
    --text-encoder="sentence-t5-base" \
    --text-embed-dim=768;

python3 src/train_ape.py \
    --experiment-name="vit_small_patch16_224_all-roberta-large-v1" \
    --image-encoder="vit_small_patch16_224" \
    --image-embed-dim=384 \
    --num-epochs=1 \
    --saving=False \
    --text-encoder="all-roberta-large-v1" \
    --text-embed-dim=1024;
