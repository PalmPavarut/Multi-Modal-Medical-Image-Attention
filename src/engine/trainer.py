import torch
import time
import os
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from src.utils.early_stopping import EarlyStopping
from src.utils.metrics import DiceScore
from src.utils.losses import FocalTverskyLoss


class Trainer:
    def __init__(self, model, train_loader, val_loader, accelerator, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator

        self.device_model, self.optimizer = self._setup_model()

        self.criterion = FocalTverskyLoss(ALPHA=0.5, BETA=0.5)
        self.metric = DiceScore()

        self.epochs = 300
        self.writer = SummaryWriter(os.path.join(save_dir, "logs"))

        self.early_stopping = EarlyStopping(
            path=os.path.join(save_dir, "best_model.pth"),
            patience=80
        )

    def _setup_model(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=6e-5)

        train_loader, val_loader, model, optimizer = self.accelerator.prepare(
            self.train_loader,
            self.val_loader,
            self.model,
            optimizer
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        return model, optimizer

    def train(self):
        for epoch in range(self.epochs):
            self._set_seed(epoch)

            train_loss, train_dice = self._train_one_epoch(epoch)
            val_loss, val_dice = self._validate(epoch)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Dice/Train", train_dice, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Dice/Val", val_dice, epoch)

            self.early_stopping(val_dice, self.model, epoch)

            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    def _train_one_epoch(self, epoch):
        self.model.train()

        total_loss, total_dice = 0, 0
        count = 0

        for images_org, images, targets in self.train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(images)
            outputs = torch.nn.functional.interpolate(
                outputs,
                size=images_org.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            loss = self.criterion(outputs, targets)
            self.accelerator.backward(loss)
            self.optimizer.step()

            with torch.no_grad():
                dice = self.metric(outputs, targets)

            total_loss += loss.item()
            total_dice += dice.item()
            count += 1

        return total_loss / count, total_dice / count

    def _validate(self, epoch):
        self.model.eval()

        total_loss, total_dice = 0, 0
        count = 0

        with torch.no_grad():
            for images_org, images, targets in self.val_loader:
                outputs = self.model(images)
                outputs = torch.nn.functional.interpolate(
                    outputs,
                    size=images_org.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

                loss = self.criterion(outputs, targets)
                dice = self.metric(outputs, targets)

                total_loss += loss.item()
                total_dice += dice.item()
                count += 1

        return total_loss / count, total_dice / count

    def _set_seed(self, epoch):
        seed = 42 + epoch
        torch.manual_seed(seed)
        np.random.seed(seed)