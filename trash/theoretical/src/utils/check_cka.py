# 
# Taken from: https://github.com/jayroxis/CKA-similarity/blob/main/CKA.py  
# 
import os
import math
import torch
import json
import numpy as np
from hypnettorch.hnets import HMLP
from hypnettorch.mnets import MLP
import matplotlib.pyplot as plt

 
class CKA(object): # renamed CudaCKA to CKA
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


if __name__ == "__main__":
    image_encoders = ["vit_base_patch16_224", "deit_base_patch16_224", "swin_small_patch4_window7.ms_in22k_ft_in1k"]
    ie_num_mapping = {}
    
    for i, name in enumerate(image_encoders):
        ie_num_mapping[name] = i
    
    def encoder_name_to_prefix(name):
        entries = name.split("_")
        model = entries[0]
        version = entries[1][0]
        return model+version
    
    def within_group(cka, weights):
        N = len(weights)
        
        linear_cka_results = [[-1000.0 for _ in range(N)] for __ in range(N)]
        kernel_cka_results = [[-1000.0 for _ in range(N)] for __ in range(N)]
        
        for i in range(N):
            for j in range(N):
                lr = cka.linear_CKA(weights[i], weights[j]).cpu().item()
                kr = cka.kernel_CKA(weights[i], weights[j]).cpu().item()
                
                linear_cka_results[i][j] = lr
                kernel_cka_results[i][j] = kr
        
        output = {"linear_cka": linear_cka_results, "kernel_cka": kernel_cka_results}
        return output
                
    def across_groups(cka, weights1, weights2):
        N = len(weights1)
        M = len(weights2)
        
        linear_cka_results = [[-1000.0 for _ in range(M)] for __ in range(N)] 
        kernel_cka_results = [[-1000.0 for _ in range(M)] for __ in range(N)] 
        
        for i in range(N):
            for j in range(M):
                lr = cka.linear_CKA(weights1[i], weights2[j]).cpu().item()
                kr = cka.kernel_CKA(weights1[i], weights2[j]).cpu().item()
                
                linear_cka_results[i][j] = lr
                kernel_cka_results[i][j] = kr
        
        output = {"linear_cka": linear_cka_results, "kernel_cka": kernel_cka_results}
        return output
    
    separate_ckpts = [
        torch.load(f"/workspace/jaisidh/hyperalignment/checkpoints/naive_mapping/{encoder_name_to_prefix(name)}_st5b_one_layer_2_ls_100/ckpt_1500.pt")["model"] for name in image_encoders
    ]
    separate_weights = []
    separate_biases = []
    
    num_weight_params = 0
    num_bias_params = 0
     
    for model_ckpt in separate_ckpts:
        for k in model_ckpt.keys():
            if "weight" in k:
                separate_weights.append(model_ckpt[k].data)
                num_weight_params = model_ckpt[k].data.shape[0] * model_ckpt[k].data.shape[1]
        
        for k in model_ckpt.keys():
            if "bias" in k:
                separate_biases.append(model_ckpt[k].data.unsqueeze(-1))
                num_bias_params = model_ckpt[k].data.shape[0]
    
    N = len(separate_weights)
    cka = CKA()
    
    separate_weights_within_group = within_group(cka, separate_weights)
    separate_biases_within_group = within_group(cka, separate_biases)
    
    linear_results = [[-1000.0 for _ in range(N)] for __ in range(N)]
    kernel_results = [[-1000.0 for _ in range(N)] for __ in range(N)]
    
    for i in range(N):
        for j in range(N):
            linear_results[i][j] = (separate_weights_within_group["linear_cka"][i][j] * num_weight_params + separate_biases_within_group["linear_cka"][j][j] * num_bias_params) / (num_weight_params + num_bias_params) 
            kernel_results[i][j] = (separate_weights_within_group["kernel_cka"][i][j] * num_weight_params + separate_biases_within_group["kernel_cka"][j][j] * num_bias_params) / (num_weight_params + num_bias_params)
            
    separate_output_within_group = {"linear_cka": linear_results, "kernel_cka": kernel_results}
     
    save_folder = "../../results/cka/baseline_0"
    os.makedirs(save_folder, exist_ok=True)
    
    linear_results_np = np.array(linear_results, dtype=float)
    plt.imshow(linear_results_np, cmap="plasma")
    plt.title("Linear CKA - separate mapper weights")
    plt.savefig(os.path.join(save_folder, "linear_cka_separate_mapper_within_group.png"))
    
    kernel_results_np = np.array(kernel_results, dtype=float)
    plt.imshow(kernel_results_np, cmap="plasma")
    plt.title("Kernel CKA - separate mapper weights")
    plt.savefig(os.path.join(save_folder, "kernel_cka_separate_mapper_within_group.png"))
    
    with open(os.path.join(save_folder, "separate_within_group_cka.json"), "w") as f:
        json.dump(separate_output_within_group, f)
     
    # joint_hypnet_ckpt = torch.load("/workspace/jaisidh/hyperalignment/checkpoints/joint_mapping/hypnet_1.pt")["model"]
    # mnet = MLP(n_in=768, n_out=768, hidden=[])
    
    # hnet = HMLP(mnet.param_shapes, layers=[], num_cond_emb=3, cond_emb_dim=768)
    # hnet.load_state_dict(joint_hypnet_ckpt)
    # hnet.eval()
    
    # W = hnet(cond_id=0)   
    

