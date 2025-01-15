import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, intermediate_dims, output_dim, use_bias=True, logit_scale=100.0):
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dims = intermediate_dims # list of ints
        self.output_dim = output_dim
        self.num_layers = len(intermediate_dims) + 1

        self.layers = []
        current_dim = input_dim
        next_dims = intermediate_dims + [output_dim]

        self.logit_scale = torch.tensor(np.log(logit_scale))

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(current_dim, next_dims[i], bias=use_bias))
            current_dim = next_dims[i]

            if i != self.num_layers - 1:
                self.layers.append(nn.GELU())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x #F.normalize(x, dim=-1)


class MlpDecoder(nn.Module):
    def __init__(self, out_shape, dim, hidden_layer_factors=[4, 16]):
        super().__init__()
        self.image_dim = out_shape[0]
        self.text_dim = out_shape[1]

        self.decoder = MLP(
            dim,
            [f*dim for f in hidden_layer_factors],
            self.image_dim * self.text_dim + self.image_dim 
        )

    def forward(self, x):
        N = x.shape[0]
        x = self.decoder(x)
        weights = x[:, :-self.image_dim].view(N, self.image_dim, self.text_dim)
        biases = x[:, -self.image_dim:]
        return weights, biases


class AttentionDecoder(nn.Module):
    def __init__(self, out_shape, dim, num_layers=3, num_heads=4, expansion_factor=2):
        super().__init__()
        self.image_dim = out_shape[0]
        self.text_dim = out_shape[1]

        self.attn_layers = []
        self.conv_layers = []
        self.fc_layers = []
        self.activation = nn.GELU()

        for i in range(num_layers):
            self.attn_layers.append(nn.TransformerEncoderLayer(
                d_model=dim if i == 0 else i * self.text_dim // num_layers,
                nhead=num_heads,
                dim_feedforward=dim * expansion_factor,
                activation="gelu",
                batch_first=True
            ))

            in_channels = 1 if i == 0 else i * self.image_dim // num_layers
            out_channels = (i+1) * self.image_dim // num_layers

            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, 1, 1))

            in_dim = dim if i == 0 else i * self.text_dim // num_layers
            out_dim = (i+1) * self.text_dim // num_layers if i != num_layers - 1 else self.text_dim + 1
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
        
        self.attn_layers = nn.ModuleList(self.attn_layers)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.fc_layers = nn.ModuleList(self.fc_layers)

    
    def forward(self, x):
        x = x.unsqueeze(0)
        for i in range(len(self.attn_layers)):
            # attention is always among the N conditional embeddings
            x = self.attn_layers[i](x)

            # then permute to upsample via Conv1d
            x = x.permute(1, 0, 2)
            x = self.conv_layers[i](x)

            x = self.activation(x)
            
            # upsample the second dim via Linear
            x = self.fc_layers[i](x)

            # permute for attention again
            x = x.permute(1, 0, 2)

        # return parameters shapes [N, D2, D1] 
        params = x.permute(1, 0, 2)
        weights = params[:, :, :-1]
        biases = params[:, :, -1]
        return weights, biases
    

def test():
    decoder = AttentionDecoder((768, 384), 32, 12, 8)

    c = 0
    for p in decoder.parameters():
        c += p.numel()

    print("Num params:", c)
    x = torch.randn(10, 32)
    y = decoder(x)
    for item in y:
        print(item.shape)
    
    decoder = MlpDecoder((768, 384), 32)

    c = 0
    for p in decoder.parameters():
        c += p.numel()

    print("Num params:", c)
    x = torch.randn(10, 32)
    y = decoder(x)
    for item in y:
        print(item.shape)
    


if __name__ == "__main__":
    test()