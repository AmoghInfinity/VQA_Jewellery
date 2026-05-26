import cv2


class CameraStream:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Unable to access webcam. "
                "Check camera permissions or camera availability."
            )

    def read_frame(self):
        success, frame = self.cap.read()

        if not success:
            raise RuntimeError(
                "Failed to read frame from webcam."
            )

        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()