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
Optimize a newly initialized conditional embedding over the dataset for 1 epoch.

- [ ] Run `flexvit-384` separate mapper training.
    1. Training a 768->384 mapper takes 37 x 843.96 GFLOPs per epoch, 10 epochs take 312,265 GFLOPs.
    2. Cifar-10 eval: 46.1 (1st epoch) -- 82.5 (10th epoch)
    3. Cifar-100 eval: 10.2 (1st epoch) -- 39.0 (10th epoch)
    4. ImageNet eval: ~2.0 (1st epoch) -- 23.8 (10th epoch)

- [ ] Eval aut-decoded cond emb
    1. Optimizing the cond emb takes 74422 x 2.42 GFLOPs per epoch = 180,101 GFLOPs.
    2. Cifar-10 eval: 84.6
    3. Cifar-100 eval: 43.2
    4. ImageNet eval: 25.8
    5. Instead of using all 595k samples, lets see with just
        - 8k samples: Cifar-10: 11.6
        - 80k samples: Cifar-10: