import torch
from torch.utils._python_dispatch import TorchDispatchMode
from collections import defaultdict
from typing import List, Any
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten

def get_shape(i):
    return i.shape

def prod(x):
    res = 1
    for i in x:
        res *= i
    return res

def matmul_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop

def addmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops

def bmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop

def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    return flop

def conv_flop(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for convolution.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    transposed = inputs[6]

    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)

def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])

def conv_backward_flop(inputs: List[Any], outputs: List[Any]):
    grad_out_shape, x_shape, w_shape = [get_shape(i) for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count = 0

    if output_mask[0]:
        grad_input_shape = get_shape(outputs[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:
        grad_weight_shape = get_shape(outputs[1])
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)

    return flop_count


flop_mapping = {
    aten.mm: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
}

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x

@dataclass
class FlopStats:
    forward: Dict[str, int]
    backward: Dict[str, int]
    
    def __add__(self, other):
        return FlopStats(
            {k: self.forward.get(k, 0) + other.forward.get(k, 0) for k in set(self.forward) | set(other.forward)},
            {k: self.backward.get(k, 0) + other.backward.get(k, 0) for k in set(self.backward) | set(other.backward)}
        )
    
    def total_forward(self) -> int:
        return sum(self.forward.values())
    
    def total_backward(self) -> int:
        return sum(self.backward.values())

class AccurateFlopCounter(TorchDispatchMode):
    def __init__(self, model=None):
        self.stats = defaultdict(lambda: FlopStats({}, {}))
        self.current_module = None
        self.in_backward = False
        self._module_stack = []
        
        if model is not None:
            for name, module in dict(model.named_modules()).items():
                if len(list(module.children())) == 0:  # Only leaf modules
                    module.register_forward_pre_hook(self.enter_module(name))
                    module.register_forward_hook(self.exit_module(name))
    
    def enter_module(self, name):
        def hook(module, inputs):
            self._module_stack.append(name)
            self.current_module = name
        return hook
    
    def exit_module(self, name):
        def hook(module, inputs, outputs):
            if self._module_stack and self._module_stack[-1] == name:
                self._module_stack.pop()
            self.current_module = self._module_stack[-1] if self._module_stack else None
        return hook
    
    @contextmanager
    def track_backward_pass(self):
        self.in_backward = True
        try:
            yield
        finally:
            self.in_backward = False
    
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        
        # Only count operations we're interested in
        if func._overloadpacket in flop_mapping:
            flop_count = flop_mapping[func._overloadpacket](args, normalize_tuple(out))
            
            # Add flops to current module if we're in one
            if self.current_module is not None:
                if self.in_backward:
                    self.stats[self.current_module].backward[str(func._overloadpacket)] = \
                        self.stats[self.current_module].backward.get(str(func._overloadpacket), 0) + flop_count
                else:
                    self.stats[self.current_module].forward[str(func._overloadpacket)] = \
                        self.stats[self.current_module].forward.get(str(func._overloadpacket), 0) + flop_count
            
            # Also track operations not belonging to any module
            if not self.current_module:
                target = "unattributed_ops"
                if self.in_backward:
                    self.stats[target].backward[str(func._overloadpacket)] = \
                        self.stats[target].backward.get(str(func._overloadpacket), 0) + flop_count
                else:
                    self.stats[target].forward[str(func._overloadpacket)] = \
                        self.stats[target].forward.get(str(func._overloadpacket), 0) + flop_count
        
        return out

    def print_statistics(self):
        total_forward = 0
        total_backward = 0
        
        print("\n=== FLOP Analysis ===")
        for module_name, stats in self.stats.items():
            fwd_total = stats.total_forward()
            bwd_total = stats.total_backward()
            total_forward += fwd_total
            total_backward += bwd_total
            
            print(f"\nModule: {module_name}")
            print("Forward pass operations:")
            for op, count in stats.forward.items():
                print(f"  {op}: {count:,} FLOPs")
            print(f"Total forward FLOPs: {fwd_total:,}")
            
            print("\nBackward pass operations:")
            for op, count in stats.backward.items():
                print(f"  {op}: {count:,} FLOPs")
            print(f"Total backward FLOPs: {bwd_total:,}")
            
            print(f"\nModule total: {fwd_total + bwd_total:,} FLOPs")
        
        print("\n=== Summary ===")
        print(f"Total forward pass FLOPs:  {total_forward:,}")
        print(f"Total backward pass FLOPs: {total_backward:,}")
        print(f"Total training FLOPs:      {total_forward + total_backward:,}")
        
def count_training_flops(model, sample_input, loss_fn):
    counter = AccurateFlopCounter(model)
    # y = torch.randn(1, 128)
    z = torch.randn(1, 128)
    
    with counter:
        # Forward pass
        y = model(sample_input) #.view(128, 128)
        loss = loss_fn(y, z)
        
        # Backward pass
        with counter.track_backward_pass():
            loss.backward()
    
    counter.print_statistics()
    return counter.stats

# model = torch.nn.Linear(32, 128*128)
model = torch.nn.Linear(128, 128)
x = torch.randn(1, 128)
loss_fn = torch.nn.MSELoss()

stats = count_training_flops(model, x, loss_fn)
print(stats)

