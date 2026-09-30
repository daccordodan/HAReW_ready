import torch
import torch.nn.functional as F

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning Loss (Khosla et al., 2020).
    
    Pulls together embeddings of samples that belong to the same activity class,
    while pushing apart embeddings belonging to different classes.
    """
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes Supervised Contrastive Loss.

        Args:
            z_i: Projection embeddings for view 1, shape (batch_size, feature_dim).
            z_j: Projection embeddings for view 2, shape (batch_size, feature_dim).
            labels: Activity class labels for the batch, shape (batch_size,).
        """
        device = z_i.device
        batch_size = z_i.shape[0]

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        features = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        self_mask = 1.0 - torch.eye(2 * batch_size, device=device)
        mask = mask * self_mask

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_counts = mask.sum(1)
        pos_counts = torch.where(pos_counts == 0, torch.ones_like(pos_counts), pos_counts)

        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_counts

        loss = -mean_log_prob_pos.mean()
        return loss