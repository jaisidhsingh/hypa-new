import torch
from torch.utils.data import DataLoader

def evaluate_clip_style_model(model, dataset, device, batch_size=256):
    """
    Evaluate a CLIP-style model on a dataset.
    The result is invariant to batch size.

    Args:
        model: The CLIP-style model (linear layer or similar).
        dataset: The evaluation dataset.
        device: The device to use (e.g., 'cuda' or 'cpu').
        batch_size: The batch size for evaluation.

    Returns:
        accuracy: The evaluation accuracy (invariant to batch size).
    """
    model.eval()  # Set the model to evaluation mode
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Disable shuffling

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for image_features, text_features in val_loader:
            # Move data to the device
            image_features = image_features.float().to(device)
            text_features = text_features.float().to(device)

            # Reshape features if necessary
            batch_size = image_features.shape[0]
            image_features = image_features.view(batch_size, -1)  # Flatten if needed
            text_features = text_features.view(batch_size, -1)  # Flatten if needed

            # Compute model outputs
            mapped_text_features = model(text_features)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True)

            # Compute similarity matrix
            sim = image_features @ mapped_text_features.T  # Dot product

            # Compute labels (assumes the i-th image corresponds to the i-th text)
            labels = torch.arange(batch_size).to(device)

            # Compute accuracy
            predictions = sim.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Compute final accuracy
    accuracy = 100 * correct / total
    return accuracy

# Example dataset and model
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, feature_dim):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.image_features = torch.randn(num_samples, feature_dim)
        self.text_features = torch.randn(num_samples, feature_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.image_features[idx], self.text_features[idx]

# Create a dummy dataset
dataset = DummyDataset(num_samples=1000, feature_dim=512)

# Create a dummy linear model
model = torch.nn.Linear(512, 512).to('cpu')

x = torch.randn(256, 512)
y = torch.rand(256, 512)

z = torch.cat([model(x), model(y)], dim=0)
u = model(torch.cat([x, y], dim=0))
print(torch.equal(z, u))

# Evaluate the model with different batch sizes
batch_sizes = [2, 256, 1000]
for batch_size in batch_sizes:
    accuracy = evaluate_clip_style_model(model, dataset, device='cpu', batch_size=batch_size)
    print(f"Evaluation Accuracy (batch_size={batch_size}): {accuracy:.2f}%")
