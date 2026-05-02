import torch
import torch.nn as nn


def _flatten(inputs, targets):
    inputs = inputs.reshape(inputs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    return inputs, targets


def _apply_sigmoid_if_needed(inputs, from_logits):
    return torch.sigmoid(inputs) if from_logits else inputs


class DiceScore(nn.Module):
    def __init__(self, smooth=1.0, from_logits=False, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = _apply_sigmoid_if_needed(inputs, self.from_logits)
        inputs, targets = _flatten(inputs, targets)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        if self.reduction == "mean":
            return dice.mean()
        elif self.reduction == "none":
            return dice
        else:
            raise ValueError("Invalid reduction")


class IoUScore(nn.Module):
    def __init__(self, smooth=1.0, from_logits=False, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = _apply_sigmoid_if_needed(inputs, self.from_logits)
        inputs, targets = _flatten(inputs, targets)

        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        if self.reduction == "mean":
            return iou.mean()
        elif self.reduction == "none":
            return iou
        else:
            raise ValueError("Invalid reduction")


class ClassificationMetrics(nn.Module):
    def __init__(self, smooth=1.0, from_logits=False, threshold=0.5):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
        self.threshold = threshold

    def forward(self, inputs, targets):
        if self.from_logits:
            inputs = torch.sigmoid(inputs)

        # 🔹 Save soft version BEFORE thresholding
        inputs_soft = inputs

        # 🔹 Flatten (batch-wise)
        inputs_flat = inputs.view(inputs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        # 🔹 Hard predictions for classification metrics
        inputs_bin = (inputs_flat > self.threshold).float()

        TP = (inputs_bin * targets_flat).sum(dim=1)
        TN = ((1 - targets_flat) * (1 - inputs_bin)).sum(dim=1)
        FP = ((1 - targets_flat) * inputs_bin).sum(dim=1)
        FN = (targets_flat * (1 - inputs_bin)).sum(dim=1)

        acc = (TP + TN + self.smooth) / (TP + TN + FP + FN + self.smooth)
        precision = (TP + self.smooth) / (TP + FP + self.smooth)
        recall = (TP + self.smooth) / (TP + FN + self.smooth)
        iou = (TP + self.smooth) / (TP + FP + FN + self.smooth)

        # DICE (global, soft, flattened)
        inputs_global = inputs_soft.view(-1)
        targets_global = targets.view(-1)

        intersection = (inputs_global * targets_global).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs_global.sum() + targets_global.sum() + self.smooth
        )

        return {
            "accuracy": acc.mean(),
            "precision": precision.mean(),
            "recall": recall.mean(),
            "iou": iou.mean(),
            "dice": dice
        }

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count