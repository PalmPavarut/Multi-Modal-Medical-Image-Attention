import torch
import torch.nn as nn
from .fusion import ModalAttentionFusion


class MultiModalSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder, fusion_module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.fusion = fusion_module

    def forward(self, x_batch):
        """
        x_batch: List[Tensor]
            Each element = modalities for one sample
        """

        batch_features = []

        for x in x_batch:
            features = self.backbone(x).feature_maps
            fused_features = self.fusion(features)
            batch_features.append(fused_features)

        # transpose batch: [(stage1,...), (stage1,...)] → [(B, ...), ...]
        aggregated = tuple(
            torch.cat([batch_features[i][s] for i in range(len(batch_features))], dim=0)
            for s in range(len(batch_features[0]))
        )

        out = self.decoder(aggregated)
        return torch.sigmoid(out)