# c=32, mlp decoder, no output norm, no init scaling, no scheduler
# num_ie = 12
python3 learn_multi_mapper.py \
    --experiment-name="ie_12_mlp_c_32_norm" \
    --num-image-encodes=12 \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --scheduler="off" \
    --normalize-output=False \
    --rescale-factor=0.0;

# c=32, mlp decoder, no output norm, no init scaling, no scheduler
# num_ie = 30
python3 learn_multi_mapper.py \
    --experiment-name="ie_30_mlp_c_32" \
    --num-image-encodes=30 \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --scheduler="off" \
    --normalize-output=False \
    --rescale-factor=0.0;
