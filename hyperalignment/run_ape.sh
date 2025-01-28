python3 src/train_ape.py \
    --experiment-name="three-fourth_vits_default_ape" \
    --image-encoder="vit_small_patch16_224" \
    --image-embed-dim=384 \
    --num-epochs=1 \
    --saving=False \
    --text-encoder="sentence-t5-base" \
    --text-embed-dim=768 \
    --data-scaling=0.75;

python3 src/train_ape.py \
    --experiment-name="one-fourth_vits_default_ape" \
    --image-encoder="vit_small_patch16_224" \
    --image-embed-dim=384 \
    --num-epochs=1 \
    --saving=False \
    --text-encoder="sentence-t5-base" \
    --text-embed-dim=768 \
    --data-scaling=0.25;

python3 src/train_ape.py \
    --experiment-name="half_vits_default_ape" \
    --image-encoder="vit_small_patch16_224" \
    --image-embed-dim=384 \
    --num-epochs=1 \
    --saving=False \
    --text-encoder="sentence-t5-base" \
    --text-embed-dim=768 \
    --data-scaling=0.5;

# python3 src/train_ape.py \
#     --experiment-name="vit_small_patch16_224_all-MiniLM-L12-v2" \
#     --image-encoder="vit_small_patch16_224" \
#     --image-embed-dim=384 \
#     --num-epochs=1 \
#     --saving=False \
#     --text-encoder="all-MiniLM-L12-v2" \
#     --text-embed-dim=384;

# python3 src/train_ape.py \
#     --experiment-name="vit_small_patch16_224_sentence-t5-base" \
#     --image-encoder="vit_small_patch16_224" \
#     --image-embed-dim=384 \
#     --num-epochs=1 \
#     --saving=False \
#     --text-encoder="sentence-t5-base" \
#     --text-embed-dim=768;

# python3 src/train_ape.py \
#     --experiment-name="vit_small_patch16_224_all-roberta-large-v1" \
#     --image-encoder="vit_small_patch16_224" \
#     --image-embed-dim=384 \
#     --num-epochs=1 \
#     --saving=False \
#     --text-encoder="all-roberta-large-v1" \
#     --text-embed-dim=1024;
