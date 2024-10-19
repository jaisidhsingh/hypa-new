# Latest repository for project

- [ ] Weight scaling during initialisation
- [ ] Image-Net default
- [x] Fix ckpt paths in `hyperalignment/src/run_eval.py`

## Hyper-net info

### `learn_multi_mapper.py`
1. Batch size = 512
2. learning rate = 1e-2
3. Dataset = CC-595k
4. Num encoders = 30
5. FLOPs for 1 training epoch = 73151.62 GFlops (unaffected by weight-normalization)
