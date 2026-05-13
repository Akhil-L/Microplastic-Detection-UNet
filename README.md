#  Microplastic Detection using U-Net

A deep learning-based semantic segmentation pipeline for detecting microplastics in microscopy images using a custom U-Net architecture built with TensorFlow and OpenCV.

---

##  Features

- U-Net based semantic segmentation model  
- Custom TensorFlow/Keras data pipeline  
- Binary mask prediction for microplastic detection  
- Class imbalance analysis and handling  
- Weighted Binary Cross-Entropy loss implementation  
- Prediction visualization and threshold analysis  

---

##  Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Google Colab  

---

##  Dataset

Used Kaggle microplastic segmentation datasets containing:
- microscopy images
- binary segmentation masks

Dataset includes:
- training set
- validation set
- test set

---

##  Workflow

### 1. Dataset Preparation
- Downloaded datasets using Kaggle API
- Verified image-mask pairing
- Structured train, validation, and test directories

### 2. Data Preprocessing
- Resized images and masks to `128x128`
- Normalized image pixel values
- Converted masks into binary format
- Built custom TensorFlow `Sequence` generator

### 3. U-Net Architecture
Implemented:
- encoder-decoder architecture
- skip connections
- sigmoid output layer for segmentation masks

### 4. Model Training
Initially trained using:
- Binary Cross-Entropy loss
- Adam optimizer

### 5. Class Imbalance Debugging
Observed model collapse into fully black predictions.

Performed dataset analysis:
```python
Average positive pixel ratio: 0.23%
Suggested positive class weight: 426
```

Implemented custom weighted BCE loss to improve minority pixel learning.

### 6. Prediction Analysis
Generated segmentation outputs using multiple thresholds:
- 0.3
- 0.5
- 0.7

Visualized:
- original images
- ground truth masks
- predicted segmentation masks

---

##  Results

After applying weighted BCE loss:
- training loss decreased steadily
- validation performance improved
- model stopped predicting empty masks
- segmentation structures became visible in predictions

### Training Progress
- Accuracy: `0.03 → 0.39`
- Validation Accuracy: `0.05 → 0.44`
- Loss: `2.22 → 1.55`

---

##  Key Learning

The biggest challenge was handling severe class imbalance in semantic segmentation tasks. The project demonstrated how models can silently fail by predicting only background pixels and how weighted loss functions help improve learning for minority classes.

---

##  Future Improvements

- Dice + Weighted BCE hybrid loss
- Attention U-Net architecture
- Higher resolution training
- IoU and Dice score evaluation
- Data augmentation techniques
- Patch-based segmentation

---

##  Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Notebook
Open:
```bash
Microplastic_Detection_UNet.ipynb
```

in Google Colab or Jupyter Notebook.

---

##  Notes

- Accuracy alone is misleading for highly imbalanced segmentation datasets.
- Positive microplastic pixels occupied less than 1% of the dataset.
- Weighted BCE significantly improved segmentation learning behavior.

---
