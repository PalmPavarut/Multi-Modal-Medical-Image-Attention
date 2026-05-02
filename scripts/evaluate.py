import argparse
import os
import torch

from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.datasets.dataset import MultiModalDataset
from src.datasets.collate import MultiModalCollator
from src.datasets.transforms import ComposePaired, Resize, ToTensor

from src.models.build import build_model
from src.engine.evaluator import Evaluator

from src.utils.metrics import ClassificationMetrics
from src.utils.losses import FocalTverskyLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    return parser.parse_args()


def build_dataloader(args):
    transform = ComposePaired([
        Resize((224, 448)),
        ToTensor(),
    ])

    dataset = MultiModalDataset(
        csv_file=args.csv,
        main_modality=args.modality,
        transform=transform
    )

    collate_fn = MultiModalCollator()

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    return dataloader


def main():
    args = parse_args()
    accelerator = Accelerator()

    dataloader = build_dataloader(args)
    model = build_model()

    # ✅ Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    model, dataloader = accelerator.prepare(model, dataloader)

    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        metric=ClassificationMetrics(from_logits=False),
        criterion=FocalTverskyLoss(),
        accelerator=accelerator
    )

    val_loss, val_metrics = evaluator.evaluate()

    if accelerator.is_local_main_process:
        print("\n===== Evaluation Results =====")
        print(f"Loss: {val_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()