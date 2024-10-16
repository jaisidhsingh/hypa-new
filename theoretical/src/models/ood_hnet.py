import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm

from hypnettorch.hnets import HMLP
from hypnettorch.mnets import MLP

from .ghn3 import from_pretrained, Graph
from .node_importance import node_selection


@torch.no_grad()
def embed_image_encoders(args, input_image_encoders):
    output = []
    names = []

    ghn3 = from_pretrained(args.ghn3_name)
    bar = tqdm(total=len(input_image_encoders))

    for model_name in input_image_encoders:
        model = timm.create_model(model_name, pretrained=False, num_classes=0)

        graph = Graph(model)
        A = graph._Adj.float()

        out = ghn3(
            model, graph,
            bn_track_running_stats=False,
            keep_grads=False,
            take_before_gnn=False,
            override_output=True
        )

        out /= out.norm(dim=-1, keepdim=True)

        if args.node_downsampling == "topk":
            downsampled_out = node_selection(out, A, args.node_k)

        elif args.node_downsampling == "mean":
            downsampled_out = out.mean(dim=0).unsqueeze(0)

        elif args.node_downsampling == "sum":
            downsampled_out = out.sum(dim=0).unsqueeze(0)

        downsampled_out /= downsampled_out.norm(dim=-1, keepdim=True)
        output.append(downsampled_out)
        names.append(model_name)

        bar.update(1)
        bar.set_postfix({"model": model_name, "done": len(output)})

        del model
        del graph

    bar.close()
    return output, names


class OracleHMLP(nn.Module):
    def __init__(
            self,
            args,
            input_image_encoders,
            input_text_encoders,
            predicted_model,
            image_cond_emb_dim,
            text_cond_emb_dim,
            variable_text_encoders=False
        ):
        super().__init__()
        self.args = args
        self.image_cond_emb_dim = image_cond_emb_dim
        self.text_cond_emb_dim = text_cond_emb_dim
        self.variable_text_encoders = variable_text_encoders
        self.num_image_encoders = len(input_image_encoders)
        self.num_text_encoders = len(input_text_encoders)

        self.param_shapes = predicted_model.param_shapes
        self.total_cond_emb_dim = args.text_embed_dim


        # image conditional embeddings
        self.image_conditional_embeddings = embed_image_encoders(args, input_image_encoders)

        self.num_selected_nodes = self.image_cond_emb_dim // self.image_conditional_embeddings[0].shape[1]
        # number of nodes needed to make a vector of the dim we want
        self.image_conditional_embeddings = [item[:self.num_selected_nodes, :] for item in self.image_conditional_embeddings]
        self.image_conditional_embeddings = [item.flatten() for item in self.image_conditional_embeddings]

        # text conditional embeddings
        if not variable_text_encoders:
            # embedding layer, since only eval ID text encoders, this will be trained and should be passable
            # M text encoders, this lookup table will correspond to M x 384
            self.text_conditional_embedding_table = nn.Embedding(len(input_text_encoders), text_cond_emb_dim)
            self.text_conditional_embeddings = None

        else:
            self.text_conditional_embedding_table = None
            self.text_conditional_embeddings = embed_image_encoders(args, input_text_encoders)
            # number of nodes needed to make a vector of the dim we want
            self.num_selected_nodes = self.text_cond_emb_dim // self.text_conditional_embeddings[0].shape[1]
            self.text_conditional_embeddings = [item[:self.num_selected_nodes, :] for item in self.text_conditional_embeddings]
            self.text_conditional_embeddings = [item.flatten() for item in self.text_conditional_embeddings]

        self.hnet = HMLP(
            self.param_shapes,
            layers=[],
            no_uncond_weights=False,
            no_cond_weights=False,
            uncond_in_size=0,
            num_cond_embs=int(self.num_image_encoders * self.num_text_encoders),
            cond_in_size=self.total_cond_emb_dim
        )
        self.hnet = self.hnet.to(self.args.device)

    def train(self):
        self.hnet.train()

    def eval(self):
        self.hnet.eval()

    def _prepare_image_text_conditional_embeddings(self, image_cond_embs, text_cond_embs):
        N, M = len(image_cond_embs), len(text_cond_embs)
        cond_embs = []

        for i in range(N):
            for j in range(M):
                cond_embs.append(
                    torch.cat(
                        [
                            image_cond_embs[i].view(1, -1).to(self.args.device),
                            text_cond_embs[j].view(1, -1).to(self.args.device)
                        ],
                        dim=1
                    ).view(self.total_cond_emb_dim,)
                )

        # return the NxM conditional embeddings corresponding to each pair in the NxM pairs
        # NxM, 768 conditional embeddings corr to NxM pairs of encoders.
        cond_embs = torch.stack(cond_embs).to(self.args.device)
        return cond_embs

    def predict_weights(self):
        conditional_embeddings = None
        if not self.variable_text_encoders:
            input_indices = torch.arange(self.num_text_encoders, dtype=torch.long).to(self.args.device)
            text_conditional_embeddings = self.text_conditional_embedding_table(input_indices)

            conditional_embeddings = self._prepare_image_text_conditional_embeddings(
                self.image_conditional_embeddings,
                text_conditional_embeddings
            )
        else:
            conditional_embeddings = self._prepare_image_text_conditional_embeddings(
                self.image_conditional_embeddings,
                self.text_conditional_embeddings
            )

        # Error debugging (refer line 97)
        """
        # self.hnet.unconditional_param_shapes_ref = None
        # uncond_input, cond_input, uncond_weights, _ = self.hnet._preprocess_forward_args(
        #     uncond_input=None, cond_input=conditional_embeddings, cond_id=None, weights=None,
        #     distilled_params=None, ret_format="squeezed", condition=None
        # )
        # print(uncond_weights)
        # print(self.hnet.unconditional_param_shapes_ref)
        """

        weights_across_all_pairs = self.hnet(cond_input=conditional_embeddings)
        return weights_across_all_pairs, conditional_embeddings

