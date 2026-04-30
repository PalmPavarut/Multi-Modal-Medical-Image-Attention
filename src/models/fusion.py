import torch
import torch.nn as nn


class ModalAttentionFusion(nn.Module):
    """
    Responsible ONLY for modality fusion.
    """

    def __init__(self, dim=0):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, layer_outputs):
        """
        layer_outputs: List[Tensor]
            Each tensor shape: (M, C, H, W) or similar
        """
        fused_outputs = []

        for layer_output in layer_outputs:
            weights = self.softmax(layer_output)
            attended = layer_output * weights

            # fuse modalities
            fused = attended.sum(dim=0, keepdim=True)
            fused_outputs.append(fused)

        return fused_outputs