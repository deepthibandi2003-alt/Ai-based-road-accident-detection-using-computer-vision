Here is your **final ready-to-upload `README.md` file** ✅
👉 Just **copy & paste this into README.md in your GitHub repo** (only change your username once).

---

# 🚗 AI-Based Road Accident Detection using Computer Vision

![Visitors](https://komarev.com/ghpvc/?username=deepthi-bandi\&label=Profile%20Views\&color=blue\&style=flat)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-black)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLO](https://img.shields.io/badge/Model-YOLO-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Description

This project is a web-based application that detects road accidents from traffic videos using Computer Vision and intelligent decision-making techniques. The system processes uploaded videos frame-by-frame and identifies accidents based on object detection and motion analysis. It helps improve road safety by providing faster detection and alert generation.

---

## 🎯 Objectives

* Detect road accidents automatically
* Reduce manual monitoring effort
* Provide real-time alerts
* Improve road safety using AI

---

## ⚙️ Features

* Upload traffic video through web interface
* Frame-by-frame video processing
* Vehicle detection using YOLO
* Motion detection using MOG2
* Accident detection using decision logic
* Bounding box visualization
* Alert notification system

---

## 🧠 Algorithms Used

### 🔹 YOLO (You Only Look Once)

* Detects vehicles like cars, bikes
* High speed and accuracy

### 🔹 MOG2 (Background Subtraction)

* Detects moving objects
* Identifies abnormal motion

### 🔹 Decision Tree (Logic-Based)

* Checks conditions
* Decides accident or not
* Rule-based detection

---

## 🌳 Accident Detection Flow

```text
Start
 ↓
Vehicle Detection (YOLO)
 ↓
Motion Analysis (MOG2)
 ↓
Abnormal Motion?
   ↓
  Yes → Collision Check → 🚨 Accident Detected
  No  → Normal Movement
```

---

## 🛠️ Technologies Used

* Python
* Flask
* OpenCV
* NumPy
* SciPy
* YOLO Model

---

## 📂 Project Structure

```bash
project/
│── app.py
│── templates/
│── static/
│── uploads/
│── best.pt
│── requirements.txt
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app.py
```



---

## 🔄 Working Process

1. Upload traffic video
2. Extract video frames
3. Detect vehicles using YOLO
4. Detect motion using MOG2
5. Apply decision logic
6. Detect accident
7. Display result and alert

---

## 📸 Outputs

* Video Upload Screen
* Processing Screen
* No Accident Screen
* Accident Detection Screen
* Alert Notification Screen

---

## ✅ Advantages

* Fast detection
* Automated system
* Easy to use
* Improves safety

---

## ❌ Limitations

* Depends on video quality
* False detection possible
* Requires trained model

---

## 🔮 Future Scope

* Real-time CCTV integration
* GPS-based location tracking
* SMS/Email alert system
* Advanced deep learning models

---

