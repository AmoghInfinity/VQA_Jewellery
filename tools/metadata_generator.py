import cv2
import json
import numpy as np
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_ASSET_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "processed",
    "necklaces",
    "necklace_processed.png"
)

METADATA_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "metadata",
    "necklaces",
    "necklace_processed.json"
)


def validate_alpha_channel(image):

    if image is None:
        raise ValueError(
            "Failed to load image."
        )

    if image.shape[2] != 4:
        raise ValueError(
            "Image lacks alpha channel."
        )


def extract_visible_mask(image):

    alpha = image[:, :, 3]

    mask = cv2.threshold(
        alpha,
        1,
        255,
        cv2.THRESH_BINARY
    )[1]

    return mask


def compute_visible_region(mask):

    coordinates = cv2.findNonZero(mask)

    x, y, w, h = cv2.boundingRect(
        coordinates
    )

    return {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h)
    }


def compute_center(region):

    center_x = (
        region["x"] +
        region["width"] // 2
    )

    center_y = (
        region["y"] +
        region["height"] // 2
    )

    return {
        "center_x": int(center_x),
        "center_y": int(center_y)
    }


def compute_aspect_ratio(region):

    return round(
        region["width"] /
        region["height"],
        4
    )


def estimate_anchor_points(region):

    ornament_top = (
        region["y"] +
        int(region["height"] * 0.15)
    )

    ornament_bottom = (
        region["y"] +
        int(region["height"] * 0.90)
    )

    return {
        "ornament_top": int(ornament_top),
        "ornament_bottom": int(ornament_bottom)
    }


def ensure_output_directory(path):

    directory = os.path.dirname(path)

    os.makedirs(
        directory,
        exist_ok=True
    )


def save_metadata(metadata):

    ensure_output_directory(
        METADATA_OUTPUT_PATH
    )

    with open(
        METADATA_OUTPUT_PATH,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


def main():

    image = cv2.imread(
        PROCESSED_ASSET_PATH,
        cv2.IMREAD_UNCHANGED
    )

    validate_alpha_channel(image)

    mask = extract_visible_mask(image)

    region = compute_visible_region(mask)

    center = compute_center(region)

    aspect_ratio = compute_aspect_ratio(
        region
    )

    anchors = estimate_anchor_points(
        region
    )

    metadata = {
        "asset_name": "necklace",
        "visible_region": region,
        "center": center,
        "aspect_ratio": aspect_ratio,
        "anchors": anchors
    }

    save_metadata(metadata)

    print(
        "Metadata generation complete."
    )

    print(json.dumps(
        metadata,
        indent=4
    ))


if __name__ == "__main__":
    main()