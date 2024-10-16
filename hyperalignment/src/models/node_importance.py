import torch


def nodes_cosine_distance(node_tensor):
    n = node_tensor.shape[0]

    full_sim = node_tensor @ node_tensor.T
    off_diagonal_sim = torch.zeros(n, n-1)
    for i in range(n):
        off_diagonal_sim[i, :] = torch.cat([full_sim[i, :i], full_sim[i, i+1:]])

    sim = off_diagonal_sim.mean(dim=-1).view(n,)
    return 1 - sim

def nodes_eigen_centrality(adjacency_matrix):
    n = adjacency_matrix.shape[0]

    eigen_values, eigen_vectors = torch.linalg.eigh(adjacency_matrix)
    highest_eigen_value = eigen_values[-1]
    eigencentrality = eigen_vectors[:, -1].view(n)
    return eigencentrality

def node_selection(node_tensor, adjacency_matrix, k):
    cosine_distance = nodes_cosine_distance(node_tensor)
    eigencentrality = nodes_eigen_centrality(adjacency_matrix)

    cd_values, cd_indices = cosine_distance.topk(k=k)
    ec_values, ec_indices = eigencentrality.topk(k=k)

    # print(f"Cosine distance based selections: {cd_indices}")
    # print(f"Eigencentrality based selections: {ec_indices}")
    # print(" ")

    metric = cosine_distance + eigencentrality
    values, indices = metric.topk(k=k)
    
    # print(f"Composite metric based selections: {indices}")

    output = {
        "cosine_distance_based_selection": node_tensor[cd_indices],
        "eigen_centrality_based_selection": node_tensor[ec_indices],
        "composite_metric_based_selection": node_tensor[indices]
    }
    return node_tensor[indices]
