import torch
import time
import os
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from src.utils.early_stopping import EarlyStopping
from src.utils.metrics import ClassificationMetrics
from src.utils.losses import FocalTverskyLoss

from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, accelerator, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator

        self.device_model, self.optimizer = self._setup_model()

        # ✅ Loss
        self.criterion = FocalTverskyLoss(alpha=0.5, beta=0.5)

        # ✅ Metrics
        self.metric = ClassificationMetrics(from_logits=False)

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

            train_loss, train_metrics = self._train_one_epoch(epoch)
            val_loss, val_metrics = self._validate(epoch)

            # ✅ Logging (TensorBoard)
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)

            for k, v in train_metrics.items():
                self.writer.add_scalar(f"{k}/Train", v, epoch)

            for k, v in val_metrics.items():
                self.writer.add_scalar(f"{k}/Val", v, epoch)

            # ✅ Early stopping uses Dice
            self.early_stopping(val_metrics["dice"], self.model, epoch)

            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    def _train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_metrics = {}
        count = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch+1}",
            disable=not self.accelerator.is_local_main_process
        )

        for images_org, images, targets in pbar:
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
                metrics = self.metric(outputs, targets)

            total_loss += loss.item()

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v.item()

            count += 1

            # ✅ Update progress bar
            avg_loss = total_loss / count
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "dice": f"{metrics.get('dice', 0):.4f}"
            })

        avg_metrics = {k: v / count for k, v in total_metrics.items()}
        return total_loss / count, avg_metrics

    def _validate(self, epoch):
        self.model.eval()

        total_loss = 0
        total_metrics = {}
        count = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Val Epoch {epoch+1}",
            disable=not self.accelerator.is_local_main_process
        )

        with torch.no_grad():
            for images_org, images, targets in pbar:
                outputs = self.model(images)
                outputs = torch.nn.functional.interpolate(
                    outputs,
                    size=images_org.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

                loss = self.criterion(outputs, targets)
                metrics = self.metric(outputs, targets)

                total_loss += loss.item()

                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v.item()

                count += 1

                # ✅ Update progress bar
                avg_loss = total_loss / count
                pbar.set_postfix({
                    "val_loss": f"{avg_loss:.4f}",
                    "dice": f"{metrics.get('dice', 0):.4f}"
                })

        avg_metrics = {k: v / count for k, v in total_metrics.items()}
        return total_loss / count, avg_metrics

    def _set_seed(self, epoch):
        seed = 42 + epoch
        torch.manual_seed(seed)
        np.random.seed(seed)