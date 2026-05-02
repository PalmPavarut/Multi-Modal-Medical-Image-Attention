import random
from typing import List, Optional
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode


# =========================
# Base Interface
# =========================
class PairedTransformBase:
    def __call__(self, image, mask, aux_images):
        raise NotImplementedError


# =========================
# Compose
# =========================
class ComposePaired(PairedTransformBase):
    def __init__(self, transforms: List[PairedTransformBase]):
        self.transforms = transforms

    def __call__(self, image, mask, aux_images):
        for t in self.transforms:
            image, mask, aux_images = t(image, mask, aux_images)
        return image, mask, aux_images


# =========================
# Resize
# =========================
class Resize(PairedTransformBase):
    def __init__(self, size=(224, 448)):
        self.resize_img = v2.Resize(size=size, antialias=True)
        self.resize_mask = v2.Resize(
            size=size,
            interpolation=InterpolationMode.NEAREST  # IMPORTANT for masks
        )

    def __call__(self, image, mask, aux_images):
        image = self.resize_img(image)
        mask = self.resize_mask(mask)

        aux_images = [
            self.resize_img(img) if img is not None else None
            for img in aux_images
        ]

        return image, mask, aux_images


# =========================
# Random Horizontal Flip
# =========================
class RandomFlip(PairedTransformBase):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, aux_images):
        if random.random() < self.p:
            image = v2.functional.hflip(image)
            mask = v2.functional.hflip(mask)

            aux_images = [
                v2.functional.hflip(img) if img is not None else None
                for img in aux_images
            ]

        return image, mask, aux_images


# =========================
# Random Rotation
# =========================
class RandomRotate(PairedTransformBase):
    def __init__(self, degrees=(-5, 5)):
        self.degrees = degrees

    def __call__(self, image, mask, aux_images):
        angle = random.uniform(*self.degrees)

        image = v2.functional.rotate(image, angle)
        mask = v2.functional.rotate(
            mask,
            angle,
            interpolation=InterpolationMode.NEAREST  # CRITICAL
        )

        aux_images = [
            v2.functional.rotate(img, angle) if img is not None else None
            for img in aux_images
        ]

        return image, mask, aux_images


# =========================
# To Tensor
# =========================
class ToTensor(PairedTransformBase):
    def __init__(self, binarize_mask=True):
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.binarize_mask = binarize_mask

    def __call__(self, image, mask, aux_images):
        image = self.transform(image)
        mask = self.transform(mask)

        if self.binarize_mask:
            mask = (mask > 0.5).float()

        aux_images = [
            self.transform(img) if img is not None else None
            for img in aux_images
        ]

        return image, mask, aux_images


# =========================
# Optional: Normalize
# =========================
class Normalize(PairedTransformBase):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, aux_images):
        image = (image - self.mean) / self.std

        aux_images = [
            (img - self.mean) / self.std if img is not None else None
            for img in aux_images
        ]

        return image, mask, aux_images