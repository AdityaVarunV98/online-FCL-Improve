from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning with optional class weighting."""
    def __init__(self, temperature=0.07, contrast_mode='all', beta=0.99):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.beta = beta

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` must be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # ====== Construct mask ======
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # ====== Contrastive feature setup ======
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'proxy':
            anchor_feature = features[:, 0]
            contrast_feature = features[:, 1]
            anchor_count = 1
            contrast_count = 1
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        # ====== Compute logits ======
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # ====== Masking setup ======
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # ====== Class weighting (new) ======
        if labels is not None:
            unique_classes, inverse_indices = torch.unique(labels.squeeze().long(), return_inverse=True)
            label_counts = torch.bincount(inverse_indices)

            eff_num = 1.0 - torch.pow(self.beta, label_counts.float())
            weights = (1.0 - self.beta) / (eff_num + 1e-8)
            weights = weights / weights.sum() * len(weights)

            # Map weights back to samples
            sample_weights = weights[inverse_indices]
            weight_matrix = sample_weights * sample_weights.T

            mean_log_prob_pos = (mask * log_prob * weight_matrix).sum(1) / (mask.sum(1) + 1e-8)
            # print("\n[SupConLoss Debug Info]")
            # print(f"Classes in batch: {unique_classes.cpu().numpy().tolist()}")
            # print(f"Label counts: {label_counts.cpu().numpy().tolist()}")
            # print(f"Assigned class weights: {weights.detach().cpu().numpy().round(4).tolist()}")
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # ====== Final loss ======
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
