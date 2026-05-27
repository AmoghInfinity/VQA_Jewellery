import cv2
import numpy as np
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "processed",
    "necklaces",
    "necklace_processed.png"
)


def load_image():

    image = cv2.imread(
        IMAGE_PATH,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise ValueError(
            "Failed to load image."
        )

    return image


def extract_mask(image):

    alpha = image[:, :, 3]

    mask = cv2.threshold(
        alpha,
        1,
        255,
        cv2.THRESH_BINARY
    )[1]

    return mask


def isolate_thick_regions(mask):

    height, width = mask.shape

    image_scale = int(
        min(height, width) / 220
    )

    image_scale = max(
        3,
        image_scale
    )

    if image_scale % 2 == 0:
        image_scale += 1

    kernel_size = image_scale

    iterations = max(
        1,
        image_scale // 4
    )

    print(
        f"Kernel Size: {kernel_size}"
    )

    print(
        f"Iterations: {iterations}"
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    eroded = cv2.erode(
        mask,
        kernel,
        iterations=iterations
    )

    dilated = cv2.dilate(
        eroded,
        kernel,
        iterations=iterations
    )

    return dilated


def get_bounding_box(mask):

    coordinates = cv2.findNonZero(mask)

    if coordinates is None:

        raise ValueError(
            "No ornament region detected."
        )

    x, y, width, height = cv2.boundingRect(
        coordinates
    )

    return x, y, width, height


def draw_debug_box(
    image,
    x,
    y,
    width,
    height
):

    debug_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGRA2BGR
    )

    cv2.rectangle(
        debug_image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        3
    )

    cv2.putText(
        debug_image,
        "THICK ORNAMENT REGION",
        (x, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return debug_image


def main():

    image = load_image()

    mask = extract_mask(image)

    thick_mask = isolate_thick_regions(
        mask
    )

    x, y, width, height = get_bounding_box(
        thick_mask
    )

    print(
        f"x={x}, y={y}, "
        f"width={width}, height={height}"
    )

    debug_image = draw_debug_box(
        image,
        x,
        y,
        width,
        height
    )

    preview = cv2.resize(
        debug_image,
        (700, 700)
    )

    cv2.imshow(
        "Thickness-Based Ornament Detection",
        preview
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()