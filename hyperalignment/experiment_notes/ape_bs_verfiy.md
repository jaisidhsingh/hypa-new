# Verifying effect of batch size in APE

- Training dataset = cc595k (558k slice)
- Num epochs = 20

## 1. `bs=256` & `lr=1e-3`

- FLOPs for 1 epoch = 877.7 Billion
- ImageNet-1k top-1 accuracy = 25.45

## 2. `bs=1024` & `lr=3e-3`

- FLOPs for 1 epoch = 1536.1 Billion
- ImageNet-1k top-1 accuracy = 26.96

## 3. `bs=4096` & `le=5e-3`

- FLOPs for 1 epoch = 4161.8 Billion
- ImageNet-1k top-1 accuracy = 27.70

## 4. `bs=16384` & `lr=1e-2`

- FLOPs for 1 epoch = 14675.9 Billion
- ImageNet-1k top-1 accuracy = 27.85
