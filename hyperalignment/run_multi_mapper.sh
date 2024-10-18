# c=32, mlp decoder, no output norm, no init scaling
python3 learn_multi_mapper.py \
    --experiment-name="mlp_c_32" \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --normalize-output=False \
    --rescale-factor=0.0;

# c=32, mlp decoder, output norm, no init scaling
python3 learn_multi_mapper.py \
    --experiment-name="mlp_c_32_norm" \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --normalize-output=True \
    --rescale-factor=0.0;

# c=32, mlp decoder, output norm, init scaling
python3 learn_multi_mapper.py \
    --experiment-name="mlp_c_32_norm_init" \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=8 \
    --normalize-output=True \
    --rescale-factor=10.0;
