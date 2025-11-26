# Pose-Matrix-AI: Kinetic Vision System

**A real-time kinetic motion analysis system featuring a sci-fi HUD overlay and 3D spatial mapping, built with OpenCV and MediaPipe.**

## 🔍 Overview
**Pose-Matrix-AI** is a computer vision tool designed to analyze human movement in real-time. Unlike standard pose detectors, this system calculates a **"Kinetic Energy" index**—a dynamic metric that visualizes the intensity of physical movement through a Heads-Up Display (HUD).

It features a "Matrix-style" aesthetic and includes a synchronized 3D spatial plotter to visualize body mechanics in three-dimensional space.

## 🚀 Key Features
* **Kinetic Analysis Engine:** Real-time calculation of movement intensity based on joint velocity.
* **Cyber-HUD Overlay:** dynamic "Heads Up Display" featuring activity bars and status indicators.
* **3D Spatial Mapping:** Simultaneous visualization of the skeletal structure in a 3D coordinate system.
* **Performance Optimized:** Uses selectively updated plotting to maintain high FPS on standard CPUs.
* **Privacy Focused:** All processing is done locally on-device; no video data is sent to the cloud.

## 🛠️ Technology Stack
* **Language:** Python 3.10+
* **Core AI:** MediaPipe (Google)
* **Computer Vision:** OpenCV
* **Visualization:** Matplotlib (3D plotting), NumPy

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/karthik-k11/Pose-Matrix-AI.git](https://github.com/karthik-k11/Pose-Matrix-AI.git)
    cd Pose-Matrix-AI
    ```

2.  **Create a Virtual Environment**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage

1.  **Run the System**
    ```bash
    python matrix_ai.py
    ```

2.  **Controls**
    * Stand 3-6 feet away from the camera for full-body detection.
    * Move your body to see the **Kinetic Output** bar react to your speed.
    * Press **`q`** on your keyboard (while the camera window is focused) to exit.

## 🧩 How It Works
The system uses **MediaPipe Pose** to extract 33 skeletal landmarks. The `MotionAnalyzer` class calculates the Euclidean distance of key joints (wrists, ankles) between frames to derive a "movement delta." This delta is smoothed and mapped to the visual interface, creating a responsive activity meter.

## 📄 License
This project is open-source and available under the MIT License.