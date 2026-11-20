# Técnico Solar Boat — Autonomous Systems Recruitment (Phase 2)

**Candidate:** João Pereira  
**Department:** Autonomous Systems (Instituto Superior Técnico)  
**Stack:** ROS 2 Humble · PyTorch · C++17 (BehaviorTree.CPP v4) · HTML5 Canvas / HTTP

---

## 1. System Architecture

This repository implements an end-to-end autonomous control pipeline in **ROS 2 Humble** connecting computer vision inference (**PyTorch CNN**) to deterministic decision-making (**BehaviorTree.CPP v4**) and closed-loop kinematic actuation in **Turtlesim**.

```text
[Static Image / Web Canvas]
        │  (/vision/image or HTTP POST /predict)
        ▼
┌───────────────────────────────┐
│  vision_node (Python/PyTorch) │  ──► TurtleVisionCNN (eval + no_grad)
└───────────────────────────────┘
        │  /vision/command (std_msgs/String: FORWARD | BACKWARD | LEFT | RIGHT)
        ▼
┌───────────────────────────────┐
│ bt_executor (C++ BT.CPP v4)   │  ──► GetVisionCommand (mutex + single-shot consume)
│                               │  ──► MoveTurtle (StatefulActionNode + /turtle1/pose)
└───────────────────────────────┘
        │  /turtle1/cmd_vel (geometry_msgs/Twist)
        ▼
   [Turtlesim]