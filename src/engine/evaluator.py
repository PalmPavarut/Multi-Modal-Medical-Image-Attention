import torch
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, dataloader, metric, criterion=None, accelerator=None):
        self.model = model
        self.dataloader = dataloader
        self.metric = metric
        self.criterion = criterion
        self.accelerator = accelerator

    def evaluate(self):
        self.model.eval()

        total_loss = 0.0
        total_metrics = {}
        count = 0

        pbar = tqdm(
            self.dataloader,
            desc="Evaluating",
            disable=self.accelerator is not None and not self.accelerator.is_local_main_process
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

                # Loss (optional)
                if self.criterion is not None:
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()

                # Metrics
                metrics = self.metric(outputs, targets)

                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v.item()

                count += 1

                # Update tqdm
                pbar.set_postfix({
                    k: f"{v.item():.4f}" for k, v in metrics.items()
                })

        avg_metrics = {k: v / count for k, v in total_metrics.items()}

        if self.criterion is not None:
            avg_loss = total_loss / count
            return avg_loss, avg_metrics

        return avg_metrics