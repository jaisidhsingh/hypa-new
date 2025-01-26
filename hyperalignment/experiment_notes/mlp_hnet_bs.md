# MLP H-net (unswept)

- Training dataset = cc595k (558k slice)
- Num epochs = 20
- 12 encoders total, `encoder_batch_size=4`

## 1. `bs=256` & `lr=1e-3`

- FLOPs for 20 epochs = 587.9 Trillion
- ImageNet-1k top-1 accuracy = 21.36 

## 2. `bs=512` & `lr=1e-2`

- FLOPs for 20 epochs = 443.3 Trillion
- ImageNet-1k top-1 accuracy = 25.46

## 3. `bs=1024` & `lr=2e-2`

- FLOPs for 20 epochs = 470.2 Trillion
- ImageNet-1k top-1 accuracy = 0.1 

## 4. `bs=4096` & `le=3e-2`

- FLOPs for 20 epochs = 1185.7 Trillion
- ImageNet-1k top-1 accuracy = 22.72 

## 5. `bs=16384` & `lr=5e-2`

- FLOPs for 20 epochs = 4343.2 Trillion
- ImageNet-1k top-1 accuracy = 0.1 
