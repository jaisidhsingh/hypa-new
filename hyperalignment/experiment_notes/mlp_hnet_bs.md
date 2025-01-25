# MLP H-net (unswept)

- Training dataset = cc595k (558k slice)
- Num epochs = 20
- 12 encoders total, `encoder_batch_size=4`

## 1. `bs=256` & `lr=1e-3`

- FLOPs for 20 epochs = 179.0 Trillion
- ImageNet-1k top-1 accuracy = 

## 2. `bs=512` & `lr=1e-2`

- FLOPs for 20 epochs = 238.7 Trillion
- ImageNet-1k top-1 accuracy = 

## 3. `bs=1024` & `lr=3e-3`

- FLOPs for 20 epochs = 470.2 Trillion
- ImageNet-1k top-1 accuracy = 

## 4. `bs=4096` & `le=5e-3`

- FLOPs for 20 epochs = 1185.7 Trillion
- ImageNet-1k top-1 accuracy = 

## 5. `bs=16384` & `lr=1e-2`

- FLOPs for 20 epochs = 4343.2 Trillion
- ImageNet-1k top-1 accuracy = 
