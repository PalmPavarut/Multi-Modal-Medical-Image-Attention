import argparse
import os
import torch

from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.datasets.dataset import MultiModalDataset
from src.datasets.collate import MultiModalCollator
from src.datasets.transforms import ComposePaired, Resize, ToTensor

from src.models.build import build_model
from src.engine.inference import InferenceEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='tm')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/preds')
    return parser.parse_args()


def build_dataloader(args):
    transform = ComposePaired([
        Resize((224, 448)),
        ToTensor(),
    ])

    dataset = MultiModalDataset(
        csv_file=args.csv,
        main_modality=args.modality,
        transform=transform,
        return_metadata=True
    )

    collate_fn = MultiModalCollator(return_metadata=True)

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
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator()

    dataloader = build_dataloader(args)
    model = build_model()

    # ✅ Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    model, dataloader = accelerator.prepare(model, dataloader)

    engine = InferenceEngine(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        accelerator=accelerator
    )

    engine.run()

    if accelerator.is_local_main_process:
        print(f"Inference completed. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()