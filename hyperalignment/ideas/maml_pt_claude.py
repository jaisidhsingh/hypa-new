import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from collections import OrderedDict

class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, first_order=False):
        """
        Initialize MAML
        
        Args:
            model: Base model to be meta-learned
            inner_lr: Learning rate for inner loop adaptation
            outer_lr: Learning rate for outer loop meta-update
            first_order: If True, use first-order approximation
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
        self.first_order = first_order

    def inner_loop(self, support_x, support_y, steps=1):
        """
        Perform inner loop adaptation
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            steps: Number of gradient steps for adaptation
            
        Returns:
            adapted_state_dict: Parameters after adaptation
        """
        # Clone model parameters to avoid modifying them during inner loop
        adapted_params = OrderedDict(
            (name, param.clone()) 
            for (name, param) in self.model.named_parameters()
        )

        # Compute gradient with respect to copied parameters
        for _ in range(steps):
            logits = self.forward_with_params(support_x, adapted_params)
            inner_loss = F.cross_entropy(logits, support_y)
            
            # Manual gradient computation
            grads = torch.autograd.grad(
                inner_loss, 
                adapted_params.values(),
                create_graph=not self.first_order,  # Create graph only if using second-order
                allow_unused=True
            )
            
            # Update parameters manually
            adapted_params = OrderedDict(
                (name, param - self.inner_lr * grad if grad is not None else param)
                for ((name, param), grad) in zip(adapted_params.items(), grads)
            )
            
        return adapted_params

    def forward_with_params(self, x, params):
        """
        Forward pass using the provided parameters instead of the model's current parameters
        
        Args:
            x: Input data
            params: Parameters to use for forward pass
            
        Returns:
            Model output using provided parameters
        """
        # Cache original parameters
        orig_params = OrderedDict(
            (name, param.clone())
            for (name, param) in self.model.named_parameters()
        )
        
        # Replace parameters with provided ones
        for (name, param) in self.model.named_parameters():
            param.data = params[name]
            
        # Forward pass
        out = self.model(x)
        
        # Restore original parameters
        for (name, param) in self.model.named_parameters():
            param.data = orig_params[name]
            
        return out

    def outer_loop(self, tasks):
        """
        Perform outer loop meta-update
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
            
        Returns:
            meta_loss: Mean loss across all tasks
        """
        meta_loss = 0.0
        
        for (support_x, support_y, query_x, query_y) in tasks:
            # Inner loop adaptation
            adapted_params = self.inner_loop(support_x, support_y)
            
            # Evaluate on query set using adapted parameters
            query_logits = self.forward_with_params(query_x, adapted_params)
            task_loss = F.cross_entropy(query_logits, query_y)
            
            meta_loss += task_loss
            
        # Average loss across tasks
        meta_loss = meta_loss / len(tasks)
        
        # Outer loop update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()

# Example usage
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

# Initialize model and MAML
model = SimpleModel(input_dim=28*28, hidden_dim=64, output_dim=10)
maml = MAML(model, inner_lr=0.01, outer_lr=0.001)

# Training loop (pseudo-code)
"""
for epoch in range(num_epochs):
    tasks = sample_tasks(task_distribution)
    meta_loss = maml.outer_loop(tasks)
    print(f"Epoch {epoch}, Meta Loss: {meta_loss}")
"""

