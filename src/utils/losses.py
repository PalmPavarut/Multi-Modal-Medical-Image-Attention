import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten(inputs, targets):
    inputs = inputs.reshape(inputs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    return inputs, targets


def _apply_sigmoid_if_needed(inputs, from_logits):
    return torch.sigmoid(inputs) if from_logits else inputs


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        inputs = _apply_sigmoid_if_needed(inputs, self.from_logits)
        inputs, targets = _flatten(inputs, targets)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0, from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs) if self.from_logits else inputs

        # BCE (use logits version if needed)
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            bce = F.binary_cross_entropy(inputs, targets)

        inputs_flat, targets_flat = _flatten(inputs_sigmoid, targets)

        intersection = (inputs_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )

        dice_loss = 1 - dice.mean()

        return bce + dice_loss


class IoULoss(nn.Module):
    def __init__(self, smooth=1.0, from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        inputs = _apply_sigmoid_if_needed(inputs, self.from_logits)
        inputs, targets = _flatten(inputs, targets)

        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1.0, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        inputs = _apply_sigmoid_if_needed(inputs, self.from_logits)
        inputs, targets = _flatten(inputs, targets)

        TP = (inputs * targets).sum(dim=1)
        FP = ((1 - targets) * inputs).sum(dim=1)
        FN = (targets * (1 - inputs)).sum(dim=1)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        loss = (1 - tversky) ** self.gamma

        return loss.mean()