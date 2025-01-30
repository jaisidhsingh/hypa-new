import os
import torch
import random
from utils.check_cka import CKA
from configs.model_configs import model_configs




def per_family(dim, indices, device, train_lim=4):
    models = model_configs.ID_multi_mapper_configs[dim]
    imagenet_folder = "/home/mila/s/sparsha.mishra/scratch/hyperalignment/results/image_embeddings/icml/eval/imagenet1k"
    
    store = torch.zeros((len(models), len(indices), dim)).to(device)
    for idx, model in enumerate(models):
        path = os.path.join(imagenet_folder, f"dim_{dim}", model, "embedded_data.pt")
        features = torch.load(path)["inputs"][indices].to(device)
        store[idx] = features
    
    training_set_indices = [x for x in range(train_lim)]
    heldout_set_indices = [y for y in range(train_lim, len(models))]

    linear_cka_store = torch.zeros((len(training_set_indices), len(heldout_set_indices)))
    rbf_cka_store = torch.zeros((len(training_set_indices), len(heldout_set_indices)))
    cka = CKA(device)

    for i in training_set_indices:
        for j in heldout_set_indices:
            linear_cka_store[i][j] = cka.linear_CKA(store[i], store[j]).cpu().item()
            rbf_cka_store[i][j] = cka.kernel_CKA(store[i], store[j]).cpu().item()
    
    return {"linear": linear_cka_store, "rbf": rbf_cka_store}

def main():
    dims = [384, 768, 1024]
    random.seed(0)
    indices = [u for u in range(50000)]
    selected_indices = random.sample(indices, 1000)
    device = "cuda"

    output = {}
    for dim in dims:
        output[dim] = per_family(dim, selected_indices, device)
    
    torch.save(output, "/home/mila/s/sparsha.mishra/projects/hypa-new/hyperalignment/ckas_id_ood.pt")


if __name__ == "__main__":
    main()
