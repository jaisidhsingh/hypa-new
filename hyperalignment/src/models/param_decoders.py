import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    """
    We receive [N, D] shaped input (cond. embeddings.)
    We want to map this to [N, D2, D1]

    Layer 1: 
    a. [N, D] -> [1, N, D]
    b. apply attention
    c. [1, N, D] -> [N, 1, D]
    d. apply conv1d to get [N, D2 // L, D]
    e. apply gelu
    f. apply fc to get [N, D2 // L, D1 // L]
    """
    def __init__(self, out_shape, dim, num_heads, expansion_factor, num_layers):
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
            out_dim = (i+1) * self.text_dim // num_layers
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
        return x.permute(1, 0, 2)


def test():
    decoder = AttentionDecoder((768, 384), 32, 8, 2, 3)
    decoder.train()

    c = 0
    for p in decoder.parameters():
        c += p.numel()

    print("Num params:", c)
    x = torch.randn(10, 32)
    print(decoder(x).shape)


if __name__ == "__main__":
    test()