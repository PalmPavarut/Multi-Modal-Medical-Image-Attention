from transformers import UperNetForSemanticSegmentation
from .model import MultiModalSegmentationModel
from .fusion import ModalAttentionFusion


def build_model():
    base_model = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-swin-large",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    fusion = ModalAttentionFusion(dim=0)

    model = MultiModalSegmentationModel(
        backbone=base_model.backbone,
        decoder=base_model.decode_head,
        fusion_module=fusion
    )

    return model