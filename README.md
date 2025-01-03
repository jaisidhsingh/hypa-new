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
Optimize a newly initialized conditional embedding over the dataset for 1 epoch. Used the hypnetwork trained with 12 encoders for this.

- [x] Run `flexvit-384` separate mapper training.
    1. Training a 768->384 mapper takes 37 x 843.96 GFLOPs per epoch, 10 epochs take 312,265 GFLOPs.
    2. Cifar-10 eval: 46.1 (1st epoch) -- 82.5 (10th epoch)
    3. Cifar-100 eval: 10.2 (1st epoch) -- 39.0 (10th epoch)
    4. ImageNet eval: ~2.0 (1st epoch) -- 23.8 (10th epoch)

- [x] Eval auto-decoded cond emb
    1. Batch size = 8
    2. Learning rate = 1e-4
    3. No scheduler, works best with low batch size and low learning rate.
    4. Will tinker with more h-params now that we have a good init method (see below).
        - Cifar-10 eval: 84.6
        - Cifar-100 eval: 43.2
        - ImageNet eval: 25.8
        - FLOPs used = 74422 x 2.42 GFLOPs per epoch = 180,101 GFLOPs.
    5. Instead of using all 595k samples, lets do this with just 80k samples (10k steps) but we do `new_cond_emb = hnet.cond_embs.mean(dim=0)`
        - Cifar-10: 84.5
        - Cifar-100: 43.1
        - ImageNet eval: 25.9
        - FLOPs used = 10k x 2.42 GFLOPs = 24,200 GFLOPs
        - Efficient w.r.t 10 epoch baseline = 12.9 times
    6. If we init new cond emb as the avg of the embs of only those which belong to its dim-family, worse off results.
    7. Need to run this with hypernetwork that has seen 30 encoders, maybe that can train in even less data and FLOPs.