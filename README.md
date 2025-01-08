# Latest repository for project

## FLOPs counts

### APE

1. FLOPs per step for a 768 to 384 connector = 843.96 GFLOPs
2. FLOPs per step for a 768 to 768 connector = 1687.92 GFLOPs
3. FLOPs per step for a 1024 768 to 1024 connector = 2250.56 GFLOPs
4. Batch size = 2^14
5. Num steps = 37
6. Num epochs = 10

## Image Classification Bench

### APE

1. ViT-S/16 -- FLOPs = 312.2 1e+12

    - CIFAR-10:     90.1  
    - CIFAR-100:    51.9
    - ImageNet-1k:  25.5

2. ViT-B/16 -- FLOPs = 624.5 1e+12

    - CIFAR-10:     91.4
    - CIFAR-100:    63.0
    - ImageNet-1k:  38.9

3. ViT-L/16 -- FLOPs = 832.7 1e+12

    - CIFAR-10:     96.6
    - CIFAR-100:    71.2
    - ImageNet-1k:  40.0

### H-net

1. ViT-S/16

    - CIFAR-10:     89.8
    - CIFAR-100:    46.7
    - ImageNet-1k:  23.4

2. ViT-B/16

    - CIFAR-10:     93.6
    - CIFAR-100:    58.0
    - ImageNet-1k:  30.3

3. ViT-L/16

    - CIFAR-10:     94.3
    - CIFAR-100:    62.0
    - ImageNet-1k:  31.4


## Hyper-net info

### `learn_multi_mapper.py`

**Linear Warmup then Cosine Decay**

1. Batch size = 512
2. learning rate = 1e-2
3. Dataset = CC-595k (595375 samples)
4. Num encoders = 30
5. Warmup steps = 500

- For `num_ie=12` and 1 epoch, cost = 29260.64 GFLOPs
- For `num_ie=30` and 1 epoch, cost = 73151.62 GFLOPs -- which is exactly 2.5 * FLOPs for `num_ie=12`.

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
        - Cifar-10: 84.5 - 76.6
        - Cifar-100: 43.1 - 32.7
        - ImageNet eval: 25.9 - 14.6 
        - FLOPs used = 10k x 2.42 GFLOPs = 24,200 GFLOPs
        - Efficient w.r.t 10 epoch baseline = 12.9 times
    6. If we init new cond emb as the avg of the embs of only those which belong to its dim-family, worse off results.
    7. Need to run this with hypernetwork that has seen 30 encoders, maybe that can train in even less data and FLOPs.