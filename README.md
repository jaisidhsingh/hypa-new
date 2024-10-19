# Latest repository for project

- [ ] Weight scaling during initialisation
- [ ] Image-Net default
- [x] Fix ckpt paths in `hyperalignment/src/run_eval.py`

## Hyper-net info

### `learn_multi_mapper.py`

**Without scheduler**

1. Batch size = 512
2. learning rate = 1e-2
3. Dataset = CC-595k (595375 samples)
4. Num encoders = 30
5. FLOPs for 1 training epoch = 
    - Weight normalization OFF + Init scaling OFF = 73151.62 GFlops
    - Weight normalization ON + Init scaling OFF = 73151.62 GFlops
    - Weight normalization ON + Init scaling ON = 41492.08 GFlops -- incorrect
    - Weight normalization OFF + Init scaling ON = 

**Linear Warmup then Cosine Decay**

1. Batch size = 512
2. learning rate = 1e-2
3. Dataset = CC-595k (595375 samples)
4. Num encoders = 30
5. FLOPs for 1 training epoch = 
    - Weight normalization OFF + Init scaling OFF = 73151.62 GFlops
    - Weight normalization ON + Init scaling OFF = 73151.62 GFlops
    - Weight normalization ON + Init scaling ON = 41492.08 GFlops -- incorrect
    - Weight normalization OFF + Init scaling ON = 
