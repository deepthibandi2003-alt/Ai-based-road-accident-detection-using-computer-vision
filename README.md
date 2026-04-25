 AI-Based Road Accident Detection using Computer Vision
Project Description

This project is a web-based application that detects road accidents from traffic videos using Computer Vision and intelligent decision-making techniques. The system processes uploaded videos frame-by-frame and identifies accidents based on object detection and motion analysis. It helps improve road safety by providing faster detection and alert generation.

Objectives
Detect road accidents automatically
Reduce manual monitoring effort
Provide real-time alerts
Improve road safety using AI

Features
Upload traffic video
Frame-by-frame video processing
Vehicle detection using YOLO
Motion detection using MOG2
Accident detection using decision logic
Display results with bounding boxes
Alert generation system

Algorithms Used
🔹 YOLO (You Only Look Once)
Used for object detection
Detects vehicles like cars, bikes
High speed and accuracy
🔹 MOG2 (Background Subtraction)
Detects moving objects
Identifies motion changes
Helps in tracking abnormal movement
🔹 Decision Tree (Logical Decision Making)
Used to decide whether an accident occurred or not
Works based on conditions like:
Sudden change in motion
Vehicle collision detection
Abnormal object behavior
If conditions are TRUE → Accident Detected
Else → No Accident

Accident Detection Logic (Decision Flow)
Start
  ↓
Detect Vehicles (YOLO)
  ↓
Analyze Motion (MOG2)
  ↓
Is motion abnormal?
      ↓
     Yes --------→ Is collision detected?
                      ↓
                    Yes → 🚨 Accident Detected
                    No  → Normal Movement
      ↓
     No  → No Accident

Technologies Used
Language: Python
Framework: Flask
Libraries: OpenCV, NumPy, SciPy, YOLO
Tools: VS Code, Jupyter Notebook

Project Structure
project/
│── app.py
│── templates/
│── static/
│── uploads/
│── best.pt
│── requirements.txt

How to Run
1. Install Dependencies
pip install -r requirements.txt
2. Run Application
python app.py
3. Open in Browser
http://127.0.0.1:5000/

Working
Upload video
Extract frames
Detect vehicles using YOLO
Detect motion using MOG2
Apply decision logic (Decision Tree concept)
Detect accident
Display result + alert

Outputs
Video Upload Screen
Processing Screen
No Accident Screen
Accident Detection Screen
Alert Screen

Advantages
Fast detection
Automated system
User-friendly
Improves safety

Limitations
Depends on video quality
False positives possible
Needs proper training data

Future Scope
Real-time CCTV integration
GPS tracking
SMS alert system
Advanced AI models
