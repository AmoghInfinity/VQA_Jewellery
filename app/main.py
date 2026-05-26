import cv2
import math

from camera import CameraStream
from detectors.face_mesh import FaceMeshDetector
from utils.smoothing import SmoothValue
from utils.overlay import overlay_png


def calculate_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    return int(
        math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )
    )


def main():

    camera = CameraStream()

    detector = FaceMeshDetector()

    # Load necklace PNG
    necklace_png = cv2.imread(
        "assets/necklaces/necklace.png",
        cv2.IMREAD_UNCHANGED
    )

    # Smoothers
    smooth_x = SmoothValue(alpha=0.4)

    smooth_y = SmoothValue(alpha=0.4)

    smooth_width = SmoothValue(alpha=0.4)

    smooth_height = SmoothValue(alpha=0.4)

    important_landmarks = [
        172,  # Left lower jaw
        397,  # Right lower jaw
        152   # Chin
    ]

    landmark_labels = {
        172: "LEFT_LOWER_JAW",
        397: "RIGHT_LOWER_JAW",
        152: "CHIN"
    }

    try:

        while True:

            frame = camera.read_frame()

            frame = cv2.flip(frame, 1)

            results = detector.process_frame(frame)

            points = detector.get_landmark_points(
                frame,
                results,
                important_landmarks
            )

            for landmark_index, (x, y) in points.items():

                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    landmark_labels[landmark_index],
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # Draw jaw width line
            if 172 in points and 397 in points:

                cv2.line(
                    frame,
                    points[172],
                    points[397],
                    (255, 0, 0),
                    2
                )

                jaw_width = calculate_distance(
                    points[172],
                    points[397]
                )

                cv2.putText(
                    frame,
                    f"JAW WIDTH: {jaw_width}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            # Necklace overlay
            if (
                172 in points and
                397 in points and
                152 in points
            ):

                left_jaw = points[172]

                right_jaw = points[397]

                chin = points[152]

                # Raw values
                raw_center_x = int(
                    (left_jaw[0] + right_jaw[0]) / 2
                )

                raw_necklace_y = (
                    chin[1] - int(jaw_width * 0.15)
                )

                raw_necklace_width = int(
                    jaw_width * 2.6
                )

                raw_necklace_height = int(
                    raw_necklace_width * 0.75
                )

                # Smoothed values
                center_x = smooth_x.update(
                    raw_center_x
                )

                necklace_y = smooth_y.update(
                    raw_necklace_y
                )

                necklace_width = smooth_width.update(
                    raw_necklace_width
                )

                necklace_height = smooth_height.update(
                    raw_necklace_height
                )

                # Resize necklace PNG
                resized_necklace = cv2.resize(
                    necklace_png,
                    (
                        necklace_width,
                        necklace_height
                    )
                )

                # Overlay coordinates
                overlay_x = (
                    center_x - necklace_width // 2
                )

                overlay_y = necklace_y - 70

                # Apply overlay
                frame = overlay_png(
                    frame,
                    resized_necklace,
                    overlay_x,
                    overlay_y
                )

            cv2.imshow(
                "Virtual Jewellery Try-On",
                frame
            )

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    finally:
        camera.release()


if __name__ == "__main__":
    main()