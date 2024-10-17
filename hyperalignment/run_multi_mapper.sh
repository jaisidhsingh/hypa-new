# c=8, mlp decoder, no output norm, no init scaling
python3 learn_multi_mapper.py \
    --experiment-name="mlp_c_8" \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=8 \
    --normalize-output=False;
    --rescale-factor=0.0;

# c=8, mlp decoder, output norm, no init scaling
python3 learn_multi_mapper.py \
    --experiment-name="mlp_c_8" \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=8 \
    --normalize-output=True;

# c=32, mlp decoder, no output norm, no init scaling
python3 learn_multi_mapper.py \
    --experiment-name="mlp_c_8" \
    --hidden-layer-factors="4,16" \
    --hnet-cond-emb-dim=32;


