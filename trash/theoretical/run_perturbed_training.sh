# python3 main.py \
#     --perturbation=static \
#     --seeds=0,1,2,3,4 \
#     --use-wandb=True \
#     --exp-name="static_vits_cc595k";

python3 main.py \
    --perturbation=dynamic \
    --seeds=0,1,2,3,4 \
    --shuffle-control=3 \
    --use-wandb=True \
    --exp-name="dynamic_vits_cc595k";
