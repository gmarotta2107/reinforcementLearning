# 🤖 Robotic Arm Control: RL vs. Neural Inverse Kinematics

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

A comprehensive simulation environment designed to benchmark and compare two distinct control paradigms for robotic manipulation: **Supervised Learning (Neural Inverse Kinematics)** and **Deep Reinforcement Learning (DDPG)**.

Built on a decoupled **Client-Server architecture**, this project separates the physics simulation (Server) from the control algorithms (Client), mimicking real-world robotic deployment.

---

## 📂 Project Structure

The codebase is organized into logical modules handling simulation, data engineering, and model training.

| Component | Description | Key Files |
| :--- | :--- | :--- |
| **Simulation Core** | Manages the physics world, robot URDF models, and object interactions. | `server.py`, `robot.urdf`, `ball.urdf` |
| **Data Generation** | Scripts to drive the robot through random configurations to harvest training data. | `client_generate_dataset_*.py` |
| **Inverse Kinematics** | Supervised learning approach to predict joint angles from spatial coordinates. | `train_inverse_kinematics.py`, `client_just_follow.py` |
| **Reinforcement Learning** | DDPG agent training for autonomous goal-seeking behavior. | `train_reinforcement_learning.py`, `ddpg.py` |
| **Utilities** | Tools for data conversion, visualization, and mathematical operations. | `converter_txt_to_csv.py`, `plot_dataset_*.py` |

---

## 🚀 Getting Started

### 1. Launch the Simulation Server
The physics engine must be running before any client script can connect. Open a terminal and run:

```bash
python server.py
