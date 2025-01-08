# python3 src/ood_autodecoding.py \
#     --image-encoder="flexivit_small.300ep_in1k" \
#     --image-embed-dim=384 \
#     --hnet-ckpt-name="ie_12_mlp_c_32_norm_ec" \
#     --hnet-ckpt-num-ie=12 \
#     --learning-rate=1e-5 \
#     --save-path="x.pt" \
#     --batch-size=8;


python3 src/ood_autodecoding.py \
    --image-encoder="visformer_tiny.in1k" \
    --image-embed-dim=384 \
    --hnet-ckpt-name="ie_12_mlp_c_32_norm" \
    --hnet-ckpt-num-ie=12 \
    --learning-rate=1e-5 \
    --save-path="x.pt" \
    --batch-size=8;

python3 src/ood_autodecoding.py \
    --image-encoder="volo_d1_224.sail_in1k" \
    --image-embed-dim=384 \
    --hnet-ckpt-name="ie_12_mlp_c_32_norm" \
    --hnet-ckpt-num-ie=12 \
    --learning-rate=1e-5 \
    --save-path="x.pt" \
    --batch-size=8;