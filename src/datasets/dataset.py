import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class MultiModalDataset(Dataset):
    def __init__(
        self,
        csv_file,
        main_modality,
        all_modalities=None,
        transform=None,
        random_drop=False,
        drop_count=0,
        return_metadata=False,
    ):
        self.df = pd.read_csv(csv_file)

        self.main_modality = main_modality
        self.mask_modality = main_modality + "L"

        self.all_modalities = all_modalities or ['dc', 'ec', 'pc', 'am', 'tm']
        self.aux_modalities = [m for m in self.all_modalities if m != main_modality]

        self.transform = transform
        self.random_drop = random_drop
        self.drop_count = drop_count
        self.return_metadata = return_metadata

        # Pre-index for fast lookup
        self.index = self._build_index()

    def _build_index(self):
        """
        Build a dictionary:
        (ID, slice) -> {modality: path}
        """
        index = {}

        for _, row in self.df.iterrows():
            key = (row['ID'], row['Data'])
            modality = row['Modality']
            path = row['Data path']

            if key not in index:
                index[key] = {}

            index[key][modality] = path

        # Keep only samples with main modality
        valid_keys = [
            key for key in index
            if self.main_modality in index[key]
        ]

        return [(key, index[key]) for key in valid_keys]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        (id_name, slice_name), modal_dict = self.index[idx]

        # --- Main image ---
        image_path = modal_dict[self.main_modality]
        image = self._load_image(image_path)

        # --- Mask ---
        if self.mask_modality not in modal_dict:
            raise ValueError(f"Missing mask for {self.main_modality} at {id_name}, {slice_name}")

        mask_path = modal_dict[self.mask_modality]
        mask = self._load_mask(mask_path)

        # --- Auxiliary modalities ---
        aux_images = []
        for m in self.aux_modalities:
            if m in modal_dict:
                aux_images.append(self._load_image(modal_dict[m]))
            else:
                aux_images.append(None)

        # --- Random modality drop ---
        if self.random_drop and self.drop_count > 0:
            aux_images = self._random_drop(aux_images)

        # --- Transform ---
        if self.transform:
            image, mask, aux_images = self.transform(image, mask, aux_images)

        if self.return_metadata:
            return image, mask, aux_images, id_name, slice_name

        return image, mask, aux_images

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _load_mask(self, path):
        return Image.open(path).convert("L")

    def _random_drop(self, aux_images):
        valid_indices = [i for i, img in enumerate(aux_images) if img is not None]

        if len(valid_indices) == 0:
            return aux_images

        drop_indices = random.sample(
            valid_indices,
            min(self.drop_count, len(valid_indices))
        )

        for i in drop_indices:
            aux_images[i] = None

        return aux_images