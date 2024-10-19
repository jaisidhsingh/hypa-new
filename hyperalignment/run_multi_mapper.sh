# c=32, mlp decoder, no output norm, no init scaling
python3 learn_multi_mapper.py \
    --experiment-name="ie_30_mlp_c_32_cosine" \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --scheduler="cosine" \
    --normalize-output=False \
    --rescale-factor=0.0;

# c=32, mlp decoder, output norm, no init scaling
python3 learn_multi_mapper.py \
    --experiment-name="ie_30_mlp_c_32_norm_cosine" \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32 \
    --scheduler="cosine" \
    --normalize-output=True \
    --rescale-factor=0.0;
