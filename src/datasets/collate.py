import torch


class MultiModalCollator:
    """
    Collate function for multi-modal dataset.

    Handles:
    - Filtering missing modalities (None)
    - Stacking tensors
    - Formatting input for model
    """

    def __init__(self, return_metadata=False):
        self.return_metadata = return_metadata

    def __call__(self, batch):
        """
        batch:
            train: (image, mask, aux_images)
            test : (image, mask, aux_images, metadata...)
        """

        if self.return_metadata:
            images, masks, aux_lists, *meta = zip(*batch)
        else:
            images, masks, aux_lists = zip(*batch)

        images_batch = torch.stack(images, dim=0)
        masks_batch = torch.stack(masks, dim=0)

        multimodal_batch = [
            self._build_multimodal_tensor(img, aux)
            for img, aux in zip(images, aux_lists)
        ]

        if self.return_metadata:
            return images_batch, multimodal_batch, masks_batch, *meta

        return images_batch, multimodal_batch, masks_batch

    def _build_multimodal_tensor(self, image, aux_images):
        """
        Combine main image + auxiliary modalities
        into shape: (num_modalities, C, H, W)
        """

        tensors = [image.unsqueeze(0)]

        for aux in aux_images:
            if aux is not None:
                tensors.append(aux.unsqueeze(0))

        return torch.cat(tensors, dim=0)