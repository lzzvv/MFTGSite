import torch
import torch.nn as nn
import torch.optim as optim


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = (targets * probs) + ((1 - targets) * (1 - probs))
        modulating_factor = (1.0 - p_t) ** self.gamma
        alpha_weight = (targets * self.alpha) + ((1 - targets) * (1 - self.alpha))
        focal_loss = alpha_weight * modulating_factor * bce_loss
        return focal_loss.mean()


class ClassBalancedFocalLoss(nn.Module):

    def __init__(self, alpha=0.8, gamma=2.0, beta=0.9999):
        super(ClassBalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, logits, targets):

        # Count samples for each class
        n_pos = (targets == 1).sum().item()
        n_neg = (targets == 0).sum().item()

        # Avoid division by zero
        if n_pos == 0 or n_neg == 0:
            # Fallback to standard focal loss
            probs = torch.sigmoid(logits)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            p_t = (targets * probs) + ((1 - targets) * (1 - probs))
            modulating_factor = (1.0 - p_t) ** self.gamma
            alpha_weight = (targets * self.alpha) + ((1 - targets) * (1 - self.alpha))
            focal_loss = alpha_weight * modulating_factor * bce_loss
            return focal_loss.mean()

        # Compute effective number of samples (CVPR 2019 formula)
        effective_num_pos = (1.0 - self.beta**n_pos) / (1.0 - self.beta)
        effective_num_neg = (1.0 - self.beta**n_neg) / (1.0 - self.beta)

        # Compute class-balanced weights
        cb_weight_pos = 1.0 / effective_num_pos
        cb_weight_neg = 1.0 / effective_num_neg

        # Normalize weights
        total_weight = cb_weight_pos + cb_weight_neg
        cb_weight_pos = cb_weight_pos / total_weight
        cb_weight_neg = cb_weight_neg / total_weight

        # Compute focal loss components
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = (targets * probs) + ((1 - targets) * (1 - probs))
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Combine alpha weight with class-balanced weight
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha * cb_weight_pos, device=targets.device),
            torch.tensor((1 - self.alpha) * cb_weight_neg, device=targets.device),
        )

        # Compute final loss
        focal_loss = alpha_weight * modulating_factor * bce_loss

        return focal_loss.mean()


class CombinedLoss(nn.Module):

    def __init__(
        self,
        focal_weight=0.7,
        dice_weight=0.3,
        focal_alpha=0.25,
        focal_gamma=2.0,
        dice_smooth=1.0,
    ):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        combined = self.focal_weight * focal + self.dice_weight * dice

        return combined
