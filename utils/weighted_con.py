import torch
import torch.nn.functional as F

class WeightedContrastiveLoss:
    def __init__(self, temperature=0.1, beta=0.99):
        """
        Class-balanced supervised contrastive loss.
        Each client can have its own instance.
        """
        self.temperature = temperature
        self.beta = beta

    def __call__(self, features, labels, verbose=False):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Normalize for cosine similarity
        features = F.normalize(features, dim=1)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)

        # Mask self-comparisons
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        # Class-balanced weights
        num_classes = int(labels.max().item()) + 1
        label_counts = torch.bincount(labels.squeeze().long(), minlength=int(labels.max().item()) + 1)
        eff_num = 1.0 - torch.pow(self.beta, label_counts.float())
        weights = (1.0 - self.beta) / (eff_num + 1e-8)
        weights = weights / weights.sum() * len(weights)
        sample_weights = weights[labels.squeeze().long()]
        weight_matrix = sample_weights * sample_weights.T

        # Log-prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Weighted positive log-prob
        mean_log_prob_pos = (mask * log_prob * weight_matrix).sum(1) / (mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()


        if verbose:
            with torch.no_grad():
                print("\n[WeightedContrastiveLoss Debug Info]")
                print(f"Batch size: {features.size(0)}")
                print(f"Feature dim: {features.size(1)}")
                print(f"Num classes in batch: {num_classes}")
                print(f"Label counts: {label_counts.cpu().numpy().tolist()}")
                print(f"Class weights: {weights.detach().cpu().numpy().round(4).tolist()}")
                print(f"Mean positive pairs per sample: {mask.sum(1).mean().item():.2f}")
                print(f"Mean logit magnitude: {logits.abs().mean().item():.4f}")
                print(f"Contrastive loss: {loss.item():.4f}")
        return loss