# TOAD
    This repo strored the source code of TOAD system

---

## 📂 Datasets

### OpenTTGame
Download from: https://lab.osai.ai/

### NCUTT
Download from: https://huggingface.co/datasets/DuyNguyenTV/NCUTT  


The dataset contains 6 videos:
- C0024 and C0035 are used for validation
- The remaining videos are used for training

### Ball ROI
- 300×300 images of the ball
- Cropped from OpenTTGame and NCUTT datasets

---

## ⚙️ Installation
1. Clone the Ultralytics repository:
   ```bash
   git clone https://github.com/ultralytics/ultralytics
2. Paste the architecture of Enhanced_YOLOv11n into Ultralytics
3. Training model Enhanced_YOLO11 with OpenTTGame and NCUTT datasets
4. Training one version of Enhanced_YOLO11 just with Ball ROI
5. Setup links of input, output, checkpoints of Enhanced_YOLOv11n in toad.py
