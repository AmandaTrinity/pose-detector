import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp


class HandDetector:
    def __init__(self, running_mode=vision.RunningMode.IMAGE):
        self.caminho_model = 'models/hand_landmarker.task'  # https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

        base_options = python.BaseOptions(model_asset_path=self.caminho_model)
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def draw_landmarks(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        hand_landmark_style = drawing_styles.get_default_hand_landmarks_style()
        hand_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for hand_landmarks in hand_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_style,
                connection_drawing_spec=hand_connection_style)

        return annotated_image

    # modo IMAGE: cada chamada e independente, usada na extracao do dataset (imagens estaticas)
    def hand_detection_image(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.detector.detect(mp_image)

    # modo VIDEO: exige timestamps crescentes por instancia, usada no loop ao vivo da webcam
    def hand_detection_video(self, frame_open, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_open)
        return self.detector.detect_for_video(mp_image, timestamp_ms)
