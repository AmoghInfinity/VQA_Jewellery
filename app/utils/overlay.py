import cv2
import numpy as np


def overlay_png(background, overlay, x, y):
    """
    Overlay a transparent PNG onto a background image.
    """

    bg_height, bg_width = background.shape[:2]

    if overlay.shape[2] < 4:
        raise ValueError(
            "Overlay image does not contain alpha channel."
        )

    overlay_height, overlay_width = overlay.shape[:2]

    # Boundary check
    if x >= bg_width or y >= bg_height:
        return background

    # Clip overlay if it goes outside frame
    if x + overlay_width > bg_width:
        overlay_width = bg_width - x
        overlay = overlay[:, :overlay_width]

    if y + overlay_height > bg_height:
        overlay_height = bg_height - y
        overlay = overlay[:overlay_height]

    if overlay_width <= 0 or overlay_height <= 0:
        return background

    # Separate channels
    overlay_rgb = overlay[:, :, :3]

    alpha_mask = overlay[:, :, 3] / 255.0

    # Region of interest
    roi = background[
        y:y + overlay_height,
        x:x + overlay_width
    ]

    # Alpha blending
    for channel in range(3):
        roi[:, :, channel] = (
            alpha_mask * overlay_rgb[:, :, channel]
            + (1 - alpha_mask) * roi[:, :, channel]
        )

    background[
        y:y + overlay_height,
        x:x + overlay_width
    ] = roi

    return background