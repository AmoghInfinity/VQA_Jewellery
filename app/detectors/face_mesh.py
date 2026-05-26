import cv2
import mediapipe as mp
from pyparsing import results


class FaceMeshDetector:
    def __init__(
        self,
        static_mode=False,
        max_faces=1,
        detection_confidence=0.5,
        tracking_confidence=0.5
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        return results

    def draw_landmarks(self, frame, results):
        if not results.multi_face_landmarks:
            return frame

        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION
            )

        return frame
    def get_landmark_points(self, frame, results, landmark_indices):
        landmark_points = {}

        if not results.multi_face_landmarks:
            return landmark_points

        height, width, _ = frame.shape

        face_landmarks = results.multi_face_landmarks[0]

        for index in landmark_indices:
            landmark = face_landmarks.landmark[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            landmark_points[index] = (x, y)

        return landmark_points