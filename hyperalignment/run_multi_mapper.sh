# c=32, mlp decoder, output norm, no init scaling, no scheduler
# num_ie = 30
python3 src/learn_multi_mapper.py \
    --experiment-name="ie_30_mlp_c_32_norm" \
    --num-image-encoders=30 \
    --hnet-cond-emb-dim=32 \
    --scheduler="off" \
    --normalize-output=True \
    --hnet-decoder-type="mlp";


# c=32, attention decoder, output norm, no init scaling, no scheduler
# num_ie = 30
python3 src/learn_multi_mapper.py \
    --experiment-name="ie_30_attention_c_32_norm" \
    --num-image-encoders=30 \
    --hnet-cond-emb-dim=32 \
    --scheduler="off" \
    --normalize-output=True \
    --hnet-decoder-type="attention";