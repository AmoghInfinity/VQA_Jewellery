import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_ASSET_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "raw",
    "necklaces",
    "necklace.png"
)

PROCESSED_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "processed",
    "necklaces",
    "necklace_processed.png"
)


def validate_alpha_channel(image):

    if image is None:
        raise ValueError(
            "Failed to load image."
        )

    if len(image.shape) < 3:
        raise ValueError(
            "Image does not contain channels."
        )

    if image.shape[2] != 4:
        raise ValueError(
            "Image does not contain alpha channel."
        )


def extract_alpha_mask(image):

    alpha_channel = image[:, :, 3]

    mask = cv2.threshold(
        alpha_channel,
        1,
        255,
        cv2.THRESH_BINARY
    )[1]

    return mask


def find_bounding_box(mask):

    coordinates = cv2.findNonZero(mask)

    x, y, w, h = cv2.boundingRect(
        coordinates
    )

    return x, y, w, h


def crop_to_visible_region(image, bbox):

    x, y, w, h = bbox

    cropped = image[
        y:y + h,
        x:x + w
    ]

    return cropped


def ensure_output_directory(path):

    directory = os.path.dirname(path)

    os.makedirs(
        directory,
        exist_ok=True
    )


def main():

    image = cv2.imread(
        RAW_ASSET_PATH,
        cv2.IMREAD_UNCHANGED
    )

    validate_alpha_channel(image)

    mask = extract_alpha_mask(image)

    bbox = find_bounding_box(mask)

    cropped = crop_to_visible_region(
        image,
        bbox
    )

    ensure_output_directory(
        PROCESSED_OUTPUT_PATH
    )

    cv2.imwrite(
        PROCESSED_OUTPUT_PATH,
        cropped
    )

    print("Processing complete.")

    print(f"Bounding Box: {bbox}")

    print(
        f"Processed asset saved to:"
        f" {PROCESSED_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()