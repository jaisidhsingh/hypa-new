# c=32, mlp decoder, no output norm, no init scaling, no scheduler
# num_ie = 15
python3 learn_multi_mapper.py \
    --experiment-name="ie_15_mlp_c_32_norm" \
    --num-image-encoders=15 \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --scheduler="off" \
    --normalize-output=False \
    --rescale-factor=0.0;

# c=32, mlp decoder, no output norm, no init scaling, no scheduler
# num_ie = 18
python3 learn_multi_mapper.py \
    --experiment-name="ie_18_mlp_c_32" \
    --num-image-encoders=18 \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --scheduler="off" \
    --normalize-output=False \
    --rescale-factor=0.0;

# c=32, mlp decoder, no output norm, no init scaling, no scheduler
# num_ie = 24
python3 learn_multi_mapper.py \
    --experiment-name="ie_24_mlp_c_32" \
    --num-image-encoders=24 \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --scheduler="off" \
    --normalize-output=False \
    --rescale-factor=0.0;
