# H-Net TE Scaling

## Smaller TE (all-Mini-LM-L12-v2)

- Total = 12, encoder-batch-size = 4
- Num training epochs = 20
- FLOPs for 20 epochs = 288.166 Trillion

## Bigger TE (all-roberta-large-v1)

- Total = 12, encoder-batch-size = 4
- Num training epochs = 20
- FLOPs for 20 epochs = 546.833 Trillion

# APE Dataset scaling

## 3/4

- FLOPs for 1 epoch = 10.791 T

## 1/2

- FLOPs for 1 epoch = 7337.952 T

## 1/4

- FLOPs for 1 epoch = 3453.154 T

# H-Net num encoders scaling

- FLOPs for 20 epochs = 1108.415T
