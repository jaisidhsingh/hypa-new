# Latest repository for project

- [x] Weight scaling during initialisation
- [x] Fix ckpt paths in `hyperalignment/src/run_eval.py`
- [x] Image-Net default

## Hyper-net info

### `learn_multi_mapper.py`

**Linear Warmup then Cosine Decay**

1. Batch size = 512
2. learning rate = 1e-2
3. Dataset = CC-595k (595375 samples)
4. Num encoders = 30
5. Warmup steps = 500

- For `num_ie=12` = 29260.64 GFLOPs
- For `num_ie=30` = 73151.62 GFLOPs -- which is exactly 2.5 * FLOPs for `num_ie=12`.

## ImageNet eval

All evals done at epoch 1 for `vit_small_patch16_224`:

- `separate`: 3.16
- `ie_30_mlp_c_32`: 23.74
- `ie_30_mlp_c_32_norm`: 23.74 
- `ie_30_mlp_c_32_cosine`: 23.05 
- `ie_30_mlp_c_32_norm_cosine`: 23.06
- `ie_30_mlp_c_32_norm_cosine_init`: 22.58

## Todos for OOD Auto-decoding
- [ ] Run `flexvit-384` separate mapper training.
- [ ] Eval aut-decoded cond emb
    1. Cifar-10 eval: 84.6
    2. Cifar-100 eval: 43.2
    3. ImageNet eval: 25.8