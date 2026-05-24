# AI Defect Detection System using YOLOv2

Deep learning-based automated defect detection system for Mobile Phone Screen Glass (MPSG) using YOLOv2 object detection architecture in MATLAB.

The project demonstrates:
- computer vision,
- object detection,
- deep learning,
- industrial AI applications,
- defect classification,
- dataset preparation,
- model validation and testing.

---

# Features

- YOLOv2 object detection model
- Automated defect detection
- Mobile phone screen inspection
- Ground truth annotation processing
- Dataset splitting and preprocessing
- Validation and testing workflows
- Precision–Recall analysis
- Miss-rate and FPPI evaluation
- MATLAB deep learning implementation

---

# Defect Types Detected

The model detects:

- Oil contamination
- Scratch defects
- Stain defects

---

# Technologies Used

- MATLAB
- YOLOv2
- Deep Learning Toolbox
- Computer Vision Toolbox
- Object Detection
- Image Processing

---

# Project Structure

```bash
MPSG 300/                     # Dataset images

Results/
 ├── Figure1_GroundTruth.png
 ├── Figure2_DatasetSplit.png
 ├── Figure3_TrainingProgress.png
 ├── Figure4_DetectionExample.png
 ├── Figure5_PrecisionRecall.png
 └── Figure6_MissRateFPPI.png

gTruth.mat                    # Ground truth annotations
labelingSession.mat           # MATLAB labeling session

PGT.m                         # Dataset preparation script
YOLO.m                        # YOLOv2 training script
Validation.m                  # Validation script
Test.m                        # Testing and evaluation script

README.md
README.txt