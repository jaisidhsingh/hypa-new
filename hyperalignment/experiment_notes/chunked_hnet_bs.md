# Chunked H-net (unswept)

- Training dataset = cc595k (558k slice)
- Num epochs = 20
- 12 encoders total, `encoder_batch_size=4`
- `chunk_size=256`

## 1. `bs=256` & `lr=1e-3`

- FLOPs for 20 epochs = 179.0 Trillion
- ImageNet-1k top-1 accuracy = 18.2

## 2. `bs=512` & `lr=1e-2`

- FLOPs for 20 epochs = 238.7 Trillion
- ImageNet-1k top-1 accuracy = 16.2

## 3. `bs=1024` & `lr=3e-3`

- FLOPs for 20 epochs = 367.8 Trillion
- ImageNet-1k top-1 accuracy = 17.3

## 4. `bs=4096` & `le=5e-3`

- FLOPs for 20 epochs = 1160.0 Trillion
- ImageNet-1k top-1 accuracy = 20.5

## 5. `bs=16384` & `lr=1e-2`

- FLOPs for 20 epochs = 4336.6 Trillion
- ImageNet-1k top-1 accuracy = 10.9
