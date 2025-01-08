# # c=32, mlp decoder, output norm, no init scaling, no scheduler
# # num_ie = 12
# python3 src/learn_multi_mapper.py \
#     --experiment-name="ie_12_mlp_c_32_norm" \
#     --num-image-encoders=12 \
#     --hnet-cond-emb-dim=32 \
#     --scheduler="off" \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp";


# # c=32, mlp decoder, output norm, no init scaling, no scheduler
# # num_ie = 18
# python3 src/learn_multi_mapper.py \
#     --experiment-name="ie_18_mlp_c_32_norm" \
#     --num-image-encoders=18 \
#     --hnet-cond-emb-dim=32 \
#     --scheduler="off" \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp";


# # c=32, mlp decoder, output norm, no init scaling, no scheduler
# # num_ie = 24
# python3 src/learn_multi_mapper.py \
#     --experiment-name="ie_24_mlp_c_32_norm" \
#     --num-image-encoders=24 \
#     --hnet-cond-emb-dim=32 \
#     --scheduler="off" \
#     --normalize-output=True \
#     --hnet-decoder-type="mlp";


# c=32, mlp decoder, output norm, no init scaling, no scheduler
# num_ie = 30
python3 src/learn_multi_mapper.py \
    --experiment-name="ie_30_mlp_c_32_norm" \
    --num-image-encoders=30 \
    --hnet-cond-emb-dim=32 \
    --scheduler="off" \
    --normalize-output=True \
    --hnet-decoder-type="mlp"\
    --num-epochs=20;
