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

        # Analysis Metrics
        self.prev_landmarks = None
        self.activity_level = 0
        self.frame_counter = 0

        # 3D Plot Setup
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_3d_plot()

        def setup_3d_plot(self):
            self.ax.set_facecolor('#001100') # Dark Green/Black background
            self.ax.grid(False)
            self.ax.set_title("SPATIAL MAPPING", color='green', fontsize=10)
            plt.ion()

        def calculate_movement_intensity(self, current_landmarks):
            """Calculates how much the user moved since the last frame."""
            if self.prev_landmarks is None:
                return 0
            
            movement = 0
            # Check movement of wrists and ankles (indices 15, 16, 27, 28)
        key_joints = [15, 16, 27, 28] 
        for idx in key_joints:
            c = current_landmarks[idx]
            p = self.prev_landmarks[idx]
            dist = math.sqrt((c.x - p.x)**2 + (c.y - p.y)**2)
            movement += dist
            
        return movement * 100 # Scale up for visibility

    def draw_tech_overlay(self, image, intensity):
        """Draws the HUD interface elements."""
        h, w, _ = image.shape
        
        # Draw border corners
        l = 30 # line length
        t = 2  # thickness
        c = self.COLOR_SECONDARY
        
        # Top Left
        cv2.line(image, (10, 10), (10 + l, 10), c, t)
        cv2.line(image, (10, 10), (10, 10 + l), c, t)
        
        # Bottom Right
        cv2.line(image, (w-10, h-10), (w-10-l, h-10), c, t)
        cv2.line(image, (w-10, h-10), (w-10, h-10-l), c, t)

        # Activity Bar (Dynamic)
        bar_width = 200
        bar_height = 15
        fill = int(min(intensity * 20, bar_width))