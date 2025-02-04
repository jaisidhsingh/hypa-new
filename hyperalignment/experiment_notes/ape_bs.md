# APE

- Training dataset = cc595k (558k slice)
- Num epochs = 20
- Only one encoder-pair.

## 1. `bs=256` & `lr=1e-3`

- FLOPs for 20 epochs = 17.555 Trillion (877.7 Billion x 20)
- ImageNet-1k top-1 accuracy = 25.45

## 2. `bs=512` & `lr=1e-3`

- FLOPs for 20 epochs = 21.9 Trillion (1097.2 Billion x 20)
- ImageNet-1k top-1 accuracy = 26.27

## 3. `bs=1024` & `lr=3e-3`

- FLOPs for 20 epochs = 30.7 Trillion (1536.1 Billion x 20)
- ImageNet-1k top-1 accuracy = 26.96

## 4. `bs=4096` & `le=5e-3`

- FLOPs for 20 epochs = 83.2 Trillion (4161.8 Billion x 20)
- ImageNet-1k top-1 accuracy = 27.70

## 5. `bs=16384` & `lr=1e-2`

- FLOPs for 20 epochs = 293.5 Trillion (14675.9 Billion x 20)
- ImageNet-1k top-1 accuracy = 27.85
