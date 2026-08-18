"""
PyTorch Dataset for RGB Wound Images & Segmentation Masks.
Supports Foot Ulcer Segmentation Challenge (FUSC), Medetec, and AZH datasets.
"""

import os
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from data.utils.lab_color import rgb_to_lab, compute_erythema_index


class RGBWoundDataset(Dataset):
    """
    Multi-dataset PyTorch loader for RGB wound images and binary/multiclass masks.
    Calculates CIE L*a*b* redness (a*) and Erythema Index (EI) maps on the fly.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: Optional[str] = None,
        target_size: Tuple[int, int] = (512, 512),
        include_lab_ei: bool = True,
        transform=None
    ):
        """
        Args:
            image_dir: Directory containing input RGB wound images.
            label_dir: Optional directory containing corresponding ground truth masks.
            target_size: (H, W) target spatial dimensions.
            include_lab_ei: If True, appends a* (redness) and EI (erythema index)
                             to input image tensor -> output shape (5, H, W).
            transform: Optional spatial/color data augmentations.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.include_lab_ei = include_lab_ei
        self.transform = transform

        self.image_files = []
        if os.path.exists(image_dir):
            valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            self.image_files = sorted([
                f for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Read RGB image
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Resize image
        rgb_resized = cv2.resize(rgb, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        rgb_norm = rgb_resized.astype(np.float32) / 255.0  # H, W, 3 in [0, 1]

        if self.include_lab_ei:
            lab = rgb_to_lab(rgb_resized)
            a_star = lab[:, :, 1]  # [-128, 127] -> normalize to [-1, 1]
            a_star_norm = a_star / 128.0

            ei_map = compute_erythema_index(rgb_resized)  # EI log ratio

            # Stack channels: [R, G, B, a*, EI] -> shape (H, W, 5)
            features = np.dstack([rgb_norm, a_star_norm[:, :, None], ei_map[:, :, None]])
        else:
            features = rgb_norm

        # Convert features to CHW Tensor
        img_tensor = torch.from_numpy(features.transpose(2, 0, 1)).float()

        # Load label mask if provided
        mask_tensor = None
        if self.label_dir:
            mask_path = os.path.join(self.label_dir, img_name)
            if not os.path.exists(mask_path):
                # Try replacing extension with .png or .jpg
                base_name = os.path.splitext(img_name)[0]
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt_path = os.path.join(self.label_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        mask_path = alt_path
                        break

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_resized = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized > 127).astype(np.int64)
                    mask_tensor = torch.from_numpy(mask_binary).long()

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, mask_tensor
