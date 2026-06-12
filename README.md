# 🖐️ Rock-Paper-Scissors Online AI Game

An interactive, real-time, multiplayer Rock-Paper-Scissors game powered by **Computer Vision**, **Deep Learning**, and **Network Sockets**. The application utilizes your webcam to detect hand gestures in real-time, matching you against an online opponent running the same synchronized AI model.

---

## 🚀 Features
* **AI Hand Gesture Recognition:** Real-time inference using a custom deep learning pipeline.
* **Online Multiplayer:** Play against real opponents over a centralized server with synchronized countdowns.
* **Robust Detection:** MediaPipe framework ensures accurate hand tracking regardless of background variations.
* **Fast-Paced Gameplay:** A 3-second countdown window forces quick decisions, with automatic score tracking (+1 point per win).

## 🧠 System Architecture & Tech Stack

The game relies on a two-step pipeline for gesture recognition:
1. **Hand Landmark Detection:** MediaPipe extracts 21 key 3D hand coordinates.
2. **Gesture Classification:** A Convolutional Neural Network (CNN) classifies the landmarks into *Rock*, *Paper*, or *Scissors*.

* **Language:** Python 3.11
* **Computer Vision & Tracking:** OpenCV, MediaPipe
* **Deep Learning Framework:** TensorFlow / Keras
* **Networking:** Socket.io / WebSockets (Python `socket` or `socketio` library)
* **Dataset:** Trained on the [Kaggle Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/glushko/rock-paper-scissors-dataset) featuring diverse backgrounds and hand shapes.

---

## 🎮 Game Controls

| Key | Action |
| :--- | :--- |
| `Spacebar` | Ready up / Start the 3-second countdown |
| `Q` | Quit the game and disconnect from server |

---
