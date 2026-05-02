import argparse
import os

from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.datasets.dataset import MultiModalDataset
from src.datasets.collate import MultiModalCollator
from src.datasets.transforms import ComposePaired, Resize, RandomFlip, RandomRotate, ToTensor

from src.models.build import build_model
from src.engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='tm')
    parser.add_argument('--csv_train', type=str, required=True)
    parser.add_argument('--csv_val', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./outputs')
    return parser.parse_args()


def build_dataloaders(args):
    train_transform = ComposePaired([
        Resize((224, 448)),
        RandomFlip(),
        RandomRotate(),
        ToTensor(),
    ])

    val_transform = ComposePaired([
        Resize((224, 448)),
        ToTensor(),
    ])

    train_dataset = MultiModalDataset(
        csv_file=args.csv_train,
        main_modality=args.modality,
        transform=train_transform,
        random_drop=True,
        drop_count=1
    )

    val_dataset = MultiModalDataset(
        csv_file=args.csv_val,
        main_modality=args.modality,
        transform=val_transform
    )

    collate_fn = MultiModalCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    accelerator = Accelerator()

    train_loader, val_loader = build_dataloaders(args)
    model = build_model()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        save_dir=args.save_dir
    )

    trainer.train()


if __name__ == "__main__":
    main()