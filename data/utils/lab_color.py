"""
Color Science Utilities: CIE L*a*b* conversion & Erythema Index (EI) computation.

Mathematical formulation:
    EI(x, y) = log10(S_red(x, y)) - log10(S_green(x, y))
"""

import numpy as np
import cv2


def rgb_to_lab(rgb_img: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image (H, W, 3) with values in range [0, 255] or [0, 1]
    to CIE L*a*b* color space.

    Returns:
        np.ndarray: L*a*b* image (H, W, 3) where:
            L*: [0, 100] (Lightness)
            a*: [-128, 127] (Green-Red axis)
            b*: [-128, 127] (Blue-Yellow axis)
    """
    if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float64:
        if rgb_img.max() <= 1.0:
            rgb_uint8 = (rgb_img * 255.0).astype(np.uint8)
        else:
            rgb_uint8 = rgb_img.astype(np.uint8)
    else:
        rgb_uint8 = rgb_img.astype(np.uint8)

    lab_img = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    # OpenCV scales L to [0, 255], a to [0, 255], b to [0, 255].
    # Standardize to L* in [0, 100], a* in [-128, 127], b* in [-128, 127]
    lab_float = lab_img.astype(np.float32)
    lab_float[:, :, 0] = lab_float[:, :, 0] * (100.0 / 255.0)
    lab_float[:, :, 1] = lab_float[:, :, 1] - 128.0
    lab_float[:, :, 2] = lab_float[:, :, 2] - 128.0

    return lab_float


def compute_erythema_index(rgb_img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes local Erythema Index (EI) map from an RGB wound image.

    EI(x, y) = log10(S_red(x, y) + eps) - log10(S_green(x, y) + eps)

    Args:
        rgb_img (np.ndarray): H x W x 3 RGB image.
        eps (float): Small epsilon to prevent log(0).

    Returns:
        np.ndarray: H x W floating point map of local erythema intensity.
    """
    img_float = rgb_img.astype(np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0

    red_channel = img_float[:, :, 0]
    green_channel = img_float[:, :, 1]

    ei_map = np.log10(red_channel + eps) - np.log10(green_channel + eps)
    return ei_map


def compute_a_star_erythema(lab_img: np.ndarray) -> np.ndarray:
    """
    Extracts the a* channel from CIE L*a*b* image, representing redness.
    Higher a* values correspond to increased tissue hyper-perfusion/erythema.
    """
    return lab_img[:, :, 1]
