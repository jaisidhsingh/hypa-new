import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraHypnet(nn.Module):
    def __init__(self, param_shapes, cond_emb_size, num_cond_embs, rank):
        super().__init__()
        self.param_shapes = param_shapes
        self.cond_emb_size = cond_emb_size
        self.num_cond_embs = num_cond_embs
        self.rank = rank

        self.conditional_embeddings = nn.Embedding(num_cond_embs, cond_emb_size)
        self.cond_to_bias = nn.Linear(cond_emb_size, param_shapes[1][0])

        self.cond_to_weights_1 = nn.Linear(cond_emb_size, param_shapes[0][0] * rank)
        self.cond_to_weights_2 = nn.Linear(cond_emb_size, param_shapes[0][1] * rank)

    def forward(self, cond_id):
        device = self.cond_to_bias.weight.data.device

        if type(cond_id) == int:
            cond_id = [cond_id]
        
        # need to send inputs to device

        cond_id = torch.tensor(cond_id).long().to(device)
        cond_embs = self.conditional_embeddings(cond_id)

        outputs = []
        for i in range(len(cond_id)):
            bias = self.cond_to_bias(cond_embs[i])
            w1 = self.cond_to_weights_1(cond_embs[i])
            w2 = self.cond_to_weights_2(cond_embs[i])

            w1 = w1.view(self.param_shapes[0][0], self.rank)
            w2 = w2.view(self.param_shapes[0][1], self.rank)

            weight = w1 @ w2.T
            params = [weight, bias]

            if len(cond_id) == 1:
                outputs = params
            else:
                outputs.append(params)

        return outputs


if __name__ == "__main__":
    param_shapes = [[768, 768], [768]]
    model = LoraHypnet(param_shapes, 768, 3, 32).to("mps")
    dev = model.cond_to_bias.weight.data.device
    print(dev)
    params = model(cond_id=[0, 1, 2])

    total = 0
    for p in model.parameters():
        total += p.numel()

    print(total)

    for param in params:
        print(param[0].shape, param[1].shape)
