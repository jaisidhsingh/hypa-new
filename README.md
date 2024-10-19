# Latest repository for project

- [x] Weight scaling during initialisation
- [x] Fix ckpt paths in `hyperalignment/src/run_eval.py`
- [ ] Image-Net default

## Hyper-net info

### `learn_multi_mapper.py`

**Without scheduler**

1. Batch size = 512
2. learning rate = 1e-2
3. Dataset = CC-595k (595375 samples)
4. Num encoders = 30
5. FLOPs for 1 training epoch = 73151.62 GFlops
6. `nan` loss values when weights are scaled at init.

**Linear Warmup then Cosine Decay**

1. Batch size = 512
2. learning rate = 1e-2
3. Dataset = CC-595k (595375 samples)
4. Num encoders = 30
5. Warmup steps = 500
6. FLOPs for 1 training epoch = 73151.62 GFlops
