import os
import torch
from tqdm import tqdm
import torchvision.utils as vutils


class InferenceEngine:
    def __init__(self, model, dataloader, output_dir, accelerator=None):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.accelerator = accelerator

        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        self.model.eval()

        pbar = tqdm(
            self.dataloader,
            desc="Inference",
            disable=self.accelerator is not None and not self.accelerator.is_local_main_process
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):

                # Handle metadata or not
                if len(batch) == 3:
                    images_org, images, _ = batch
                    filenames = [f"{batch_idx}_{i}" for i in range(len(images))]
                else:
                    images_org, images, _, *meta = batch
                    filenames = self._build_filenames(meta, batch_idx)

                outputs = self.model(images)

                outputs = torch.nn.functional.interpolate(
                    outputs,
                    size=images_org.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

                preds = torch.sigmoid(outputs)

                # Save results
                self._save_predictions(preds, filenames)

    def _save_predictions(self, preds, filenames):
        """
        Save predicted masks as images
        """

        for i in range(preds.shape[0]):
            path = os.path.join(self.output_dir, f"{filenames[i]}.png")

            vutils.save_image(preds[i], path)

    def _build_filenames(self, meta, batch_idx):
        """
        Optional: build filenames from metadata
        """
        try:
            # Example: (ID, slice)
            ids = meta[0]
            slices = meta[1]

            return [f"{ids[i]}_{slices[i]}" for i in range(len(ids))]

        except:
            return [f"{batch_idx}_{i}" for i in range(len(meta[0]))]