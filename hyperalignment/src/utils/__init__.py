import torch
from .calflops.flops_counter import calculate_flops


def get_mapper_flops(model, input_shape, include_backward=True):
    exponent_map = {
        "K": 1e+3,
        "M": 1e+6,
        "G": 1e+9,
        "T": 1e+12
    }
    flops, macs, _ = calculate_flops(model=model, input_shape=input_shape, print_results=False)
    flops = flops.split(" ")
    num = float(flops[0])
    exp = exponent_map[flops[1][0]]

    if include_backward:
        return 3 * num, exp
    else:
        return num, exp


def get_hypnet_flops(hypnet, kwargs, include_backward=True):
    exponent_map = {
        "K": 1e+3,
        "M": 1e+6,
        "G": 1e+9,
        "T": 1e+12
    }
    flops, macs, _ = calculate_flops(model=hypnet, kwargs=kwargs, print_results=False)
    flops = flops.split(" ")
    num = float(flops[0])
    exp = exponent_map[flops[1][0]]

    if include_backward:
        return 3 * num, exp
    else:
        return num, exp

def get_ghn3_hypnet_flops(ghn3_hypnet, kwargs, include_backward=True):
    exponent_map = {
        "K": 1e+3,
        "M": 1e+6,
        "G": 1e+9,
        "T": 1e+12
    }
    flops, macs, _ = calculate_flops(model=ghn3_hypnet.hnet, kwargs=kwargs, print_results=False)
    flops = flops.split(" ")
    num = float(flops[0])
    exp = exponent_map[flops[1][0]]

    if include_backward:
        return 3 * num, exp
    else:
        return num, exp


def get_hausdorff_distance(x, y):
    """
    Compute the Hausdorff distance between two sets of embeddings.
    
    Args:
    x: torch.Tensor, shape: [N, D]
    y: torch.Tensor, shape: [M, D]
    
    Returns:
    float: The Hausdorff distance between x and y
    """
    # Compute pairwise distances
    pairwise_distances = torch.cdist(x, y)
    
    # Compute directed Hausdorff distances
    h_xy = torch.max(torch.min(pairwise_distances, dim=1)[0])
    h_yx = torch.max(torch.min(pairwise_distances, dim=0)[0])
    
    # Return the maximum of the two directed distances
    return torch.max(h_xy, h_yx).item()

def test_hausdorff_distance():
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    print(get_hausdorff_distance(x, y))


if __name__ == "__main__":
    test_hausdorff_distance()
