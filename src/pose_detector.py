#funcao init
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.core import optional_dependencies
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.caminho_model = 'models/pose_landmarker_lite.task' #https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

        # step 2 do colab que google disponibilizou
        base_options = python.BaseOptions(model_asset_path=self.caminho_model)
        self.options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False) # acho que false fica mais rápido
        self.detector = vision.PoseLandmarker.create_from_options(self.options)

    def draw_landmarks(self,rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image) # 'pintamos' na cópia

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        # importante para ser for mais de uma pessoa
        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)

        return annotated_image
    
    # no colab, lê a foto do disco e detect()
    # para video, a lógica é diferente
    def pose_detection(self, frame_open, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_open)
        result = self.detector.detect_for_video(mp_image, timestamp_ms) #step 4
        return result