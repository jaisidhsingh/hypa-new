python3 src/evaluation.py \
    --exp-name="vit_small_patch16_224_sentence-t5-base" \
    --seed=0 \
    --run-type="sep" \
    --epoch=1 \
    --encoder-index=0 \
    --image-embed-dim=384 \
    --text-embed-dim=768 \
    --text-encoder="sentence-t5-base" \
    --benchmarks="cifar10,cifar100,imagenet1k";


# # all-MiniLM-L12-v2
# python3 src/learn_hnet.py \
#     --experiment-name="allMiniLM_hnet_12-4_fmlp_c-32_bs-512_lr-1e-2" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=20 \
#     --learning-rate=1e-2 \
#     --batch-size=512 \
#     --random-seed=0 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp" \
#     --largest-text-dim=384 \
#     --text-embed-dim=384 \
#     --text-encoder="all-MiniLM-L12-v2";

# # all-roberta-large-v1
# python3 src/learn_hnet.py \
#     --experiment-name="roberta_hnet_12-4_fmlp_c-32_bs-512_lr-1e-2" \
#     --num-image-encoders=12 \
#     --encoder-batch-size=4 \
#     --hnet-cond-emb-dim=32 \
#     --cond-type="features" \
#     --num-epochs=20 \
#     --learning-rate=1e-2 \
#     --batch-size=512 \
#     --random-seed=0 \
#     --scheduler="off" \
#     --warmup-steps=100 \
#     --emb-loss=False \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp" \
#     --largest-text-dim=1024 \
#     --text-embed-dim=1024 \
#     --text-encoder="all-roberta-large-v1";