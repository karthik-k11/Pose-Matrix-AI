import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import math

class MotionAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Pose
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
        self.ax.set_facecolor('#001100')
        self.ax.grid(False)
        self.ax.set_title("SPATIAL MAPPING", color='green', fontsize=10)
        plt.ion()

    def calculate_movement_intensity(self, current_landmarks):
        """Calculates how much the user moved since the last frame."""
        if self.prev_landmarks is None:
            return 0
            
        movement = 0
        # Check movement of wrists and ankles 
        key_joints = [15, 16, 27, 28] 
        for idx in key_joints:
            c = current_landmarks[idx]
            p = self.prev_landmarks[idx]
            dist = math.sqrt((c.x - p.x)**2 + (c.y - p.y)**2)
            movement += dist
            
        return movement * 100

    def draw_tech_overlay(self, image, intensity):
        """Draws the HUD interface elements."""
        h, w, _ = image.shape
        
        # Draw border corners
        l = 30 
        t = 2  
        c = self.COLOR_SECONDARY
        
        # Top Left
        cv2.line(image, (10, 10), (10 + l, 10), c, t)
        cv2.line(image, (10, 10), (10, 10 + l), c, t)
        
        # Bottom Right
        cv2.line(image, (w-10, h-10), (w-10-l, h-10), c, t)
        cv2.line(image, (w-10, h-10), (w-10, h-10-l), c, t)

        # Activity Bar
        bar_width = 200
        bar_height = 15
        fill = int(min(intensity * 20, bar_width))
        
        cv2.rectangle(image, (20, h-40), (20 + bar_width, h-25), (0, 50, 0), 1)
        cv2.rectangle(image, (20, h-40), (20 + fill, h-25), self.COLOR_PRIMARY, -1)
        cv2.putText(image, f"KINETIC OUTPUT: {int(intensity*100)}%", (20, h-50), 
                    cv2.FONT_HERSHEY_PLAIN, 1, self.COLOR_SECONDARY, 1)

    def process_frame(self, frame):
        self.frame_counter += 1
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.pose_engine.process(rgb_frame)
        
        display_frame = frame.copy()
        
        # Darken the background 
        overlay = np.zeros_like(frame)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            #Calculate Logic
            curr_intensity = self.calculate_movement_intensity(landmarks)
            self.activity_level = 0.8 * self.activity_level + 0.2 * curr_intensity
            
            #Draw Skeleton
            connections = self.mp_pose.POSE_CONNECTIONS
            
            # Convert to pixel coords
            px_points = {}
            for idx, lm in enumerate(landmarks):
                px_points[idx] = (int(lm.x * w), int(lm.y * h))

            # Draw Lines
            for start_idx, end_idx in connections:
                if start_idx in px_points and end_idx in px_points:
                    pt1 = px_points[start_idx]
                    pt2 = px_points[end_idx]
                    
                    # Dynamic color based on intensity
                    line_color = self.COLOR_SECONDARY if self.activity_level < 1.5 else self.COLOR_ALERT
                    
                    cv2.line(display_frame, pt1, pt2, line_color, 1)
                    
            # Draw Nodes
            for idx, pt in px_points.items():
                cv2.rectangle(display_frame, (pt[0]-2, pt[1]-2), (pt[0]+2, pt[1]+2), self.COLOR_PRIMARY, -1)

            # 3. Update 3D Plot
            if self.frame_counter % 5 == 0:
                self.update_3d_view(landmarks)
                
            self.prev_landmarks = landmarks
            
        # Draw HUD
        self.draw_tech_overlay(display_frame, self.activity_level)
        
        return display_frame

    def update_3d_view(self, landmarks):
        self.ax.clear()
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_facecolor('#001100')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        
        # Extract coordinates
        xs = [lm.x for lm in landmarks]
        ys = [lm.z for lm in landmarks] # Swapped Y and Z for better 3D viewing angle
        zs = [-lm.y for lm in landmarks] # Inverted Y for upright orientation

        self.ax.scatter(xs, ys, zs, c='cyan', s=20, marker='s') # 's' for square marker
        
        # Draw connections manually for 3D
        for start, end in self.mp_pose.POSE_CONNECTIONS:
            self.ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], c='lime', linewidth=1)
            
        plt.draw()
        plt.pause(0.001)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 800) # Width
        cap.set(4, 600) # Height
        
        print("System Initialized. Press 'q' to terminate.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            processed_frame = self.process_frame(frame)
            cv2.imshow('KINETIC ANALYZER V1.0', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        plt.close()

if __name__ == "__main__":
    app = MotionAnalyzer()
    app.run()