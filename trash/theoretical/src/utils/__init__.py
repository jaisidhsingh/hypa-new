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
