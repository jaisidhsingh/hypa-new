import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLoss():
    def __init__(self, args):
        self.args = args
        self.device = "cpu"
        if args is not None:
            self.device = args.device

    def get_labels(self, batch_size):
        labels = torch.arange(batch_size, dtype=torch.long)
        return labels

    def compute_loss_and_accuracy(self, logit_scale, image_features, text_features):
        batch_size = image_features.shape[0]

        labels = self.get_labels(batch_size).to(self.device)
        logit_scale = logit_scale.exp().to(self.device)
        image_features = image_features.to(self.device)

        logits_per_image = logit_scale * (image_features @ text_features.T)
        correct = (logits_per_image.argmax(dim=-1) == labels).sum().item()

        logits_per_text = logit_scale * (text_features @ image_features.T)
        
        loss = F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        return loss / 2, correct


class SelfContrastiveLoss():
    def __init__(self, args):
        self.device = args.device
    
    def compute_loss(self, x):
        sim = x @ x.T
        labels = torch.arange(x.shape[0], dtype=torch.long).to(x.device)
        loss = F.cross_entropy(sim, labels)
        return loss


class JointClipLoss():
    def __init__(self, args):
        self.args = args
        self.device = args.device
    
    def get_labels(self, B, N, M):
        coeffs = torch.arange(B, dtype=torch.long)
        labels = torch.ones(B, N, M)
        for i in range(B):
            labels[i] = coeffs[i] * labels[i]
        
        return labels.long().to(self.device)

    def compute_loss_and_accuracy(self, logit_scales, image_features, text_features):
        [B, N, M, D] = text_features.shape
        
        labels = self.get_labels(B, N, M)
        # initially logit_scales has shape [N, M]
        logit_scales = logit_scales.exp().repeat(B, 1, 1).view(B, 1, N, M)
        
        sim = torch.einsum("bnd,xnmd->bxnm", image_features, text_features)
        logits_per_image = logit_scales * sim
        logits_per_text = sim.permute(1, 0, 2, 3)
        
        preds = sim.argmax(dim=1)
        correct = (preds == labels).sum().item()
        
        loss = F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        return loss / 2, correct
