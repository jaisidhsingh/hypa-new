from pylab import plt
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_cost_matrix(x1, x2):
    x1 = x1 - x1.mean(axis=0, keepdims=True)
    x2 = x2 - x2.mean(axis=0, keepdims=True)
    
    norm1 = np.linalg.norm(x1, axis=0, keepdims=True)
    norm2 = np.linalg.norm(x2, axis=0, keepdims=True)
    x1_norm = x1 / (norm1 + 1e-8)
    x2_norm = x2 / (norm2 + 1e-8)
    
    correlation_matrix = x1_norm.T @ x2_norm
    cost_matrix = 1 - correlation_matrix
    
    return cost_matrix

def find_best_permutation(ood_model_fts, id_models_fts_list):
    best_cost = float('inf')
    best_perm = None
    
    for fts in id_models_fts_list:
        cost_matrix = compute_cost_matrix(ood_model_fts, fts)
        r, c = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[r, c].sum()
        
        if total_cost < best_cost:
            best_cost = total_cost

            P = np.zeros_like(cost_matrix)
            P[r, c] = 1
            best_perm = P
            
    print(best_cost)
    print(best_perm.sum())
    plt.imshow(best_perm)
    plt.savefig("x.png")

    return best_perm

def align_features(ood_model_fts, id_models_fts_list):
    best_perm = find_best_permutation(ood_model_fts, id_models_fts_list)
    
    aligned_features = ood_model_fts @ best_perm
    return aligned_features

