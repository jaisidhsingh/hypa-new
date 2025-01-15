import sys
import numpy as np
from torch.utils.flop_counter import FlopCounterMode

from models import *
from training.loss_functions import ClipLoss
from configs.model_configs import model_configs


def count_hnet_flops(bs, include_backward=True):
    kwargs = model_configs.hnet_decoder_configs["mlp"]

    model =  ConditionalHyperNetwork(
        [[1024, 768], [1024]], cond_emb_dim=32,
        num_cond_embs=12, image_embed_dims=[384,768,1024], kwargs=kwargs 
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(bs, 32)
    y = torch.randn(512, bs, 384)
    z = torch.randn(512, 768)
    logit_scale = torch.tensor(np.log(100.0))
    flop_counter = FlopCounterMode(model, display=True, depth=4)

    with flop_counter:
        for i in range(1):
            if include_backward:
                opt.zero_grad()
                weights, biases = model(cond_id=x, image_embed_dim=384, normalize_output=True, nolookup=True)

                mapped_text_features = model.map_features(weights, biases, z)
                loss, corrects = model.compute_loss(logit_scale, y, mapped_text_features, emb_loss=False)

                loss.backward()
                opt.step()
            else:
                weights, biases = model(cond_id=x, image_embed_dim=384, normalize_output=True, nolookup=True)


def count_ape_flops(bs, include_backward=True):
    model = MLP(768, [], 384)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(bs, 768)
    y = torch.randn(bs, 384)
    logit_scale = torch.tensor(np.log(100.0))
    flop_counter = FlopCounterMode(model, display=True, depth=4)
    criterion = ClipLoss(None)

    with flop_counter:
        for i in range(1):
            if include_backward:
                opt.zero_grad()
                mapped_text_features = model(x)
                loss, corrects = criterion.compute_loss_and_accuracy(logit_scale, y, mapped_text_features)

                loss.backward()
                opt.step()
            else:
                mapped_text_features = model(x)

ib = sys.argv[3] != "fwd"

if sys.argv[1] == "ape":
    count_ape_flops(int(sys.argv[2]), ib)

elif sys.argv[1] == "hnet":
    count_hnet_flops(int(sys.argv[2]), ib)