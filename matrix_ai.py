import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import math 

class MotionAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose_engine = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, 
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Constants for Visuals
        self.COLOR_PRIMARY = (0, 255, 255)   # Cyan
        self.COLOR_SECONDARY = (0, 255, 0)   # Matrix Green
        self.COLOR_ALERT = (0, 0, 255)       # Red