# From DCT Features to CNNs: MNIST Classification

**Project Type:** Machine Learning / Computer Vision  
**Author:** Sadia Rahman  
**Focus:** Feature Engineering, Dimensionality Reduction, and Model Comparison  

## Description

This project builds an end-to-end image classification pipeline on the MNIST handwritten digit dataset. It combines signal-based feature engineering using the Discrete Cosine Transform (DCT) with classical machine learning models and deep learning.

The goal is to compare handcrafted feature representations against learned representations, and evaluate tradeoffs in accuracy, interpretability, and model complexity.

## Dataset

- MNIST Handwritten Digit Dataset  
- 70,000 grayscale images (28×28)  
- Subsampled to 2,000 images (200 per digit) for efficiency  

## Pipeline Overview

1. Image Reshaping  
2. DCT Feature Extraction  
3. Directional Masking  
4. Eigen Decomposition (PCA-style)  
5. Classification (RF, SVM, CNN)  
6. Model Comparison  

## Data Preview

<img width="644" height="256" alt="image" src="https://github.com/user-attachments/assets/fe899307-1992-4bc6-98ad-155cebc94389" />

## Feature Engineering

### Discrete Cosine Transform (DCT)

Each image is transformed into the frequency domain to capture global structure instead of raw pixel values.

### Directional Masks

Three masks extract structured frequency information:

- Horizontal  
- Vertical  
- Diagonal  

### Dimensionality Reduction

- Top 20 eigenfeatures per direction  
- Final feature vector:

## Models

### Random Forest

Trained on the 60-dimensional DCT feature set.

<img width="535" height="463" alt="image" src="https://github.com/user-attachments/assets/3e2d0c23-02b1-40fc-b1eb-6f69dc3720a2" />

**Results:**
- Train Accuracy: 1.00  
- Test Accuracy: 0.8975  

### Custom SVM (QP Solver)

Implemented from scratch using quadratic programming and one-vs-rest classification.

<img width="491" height="448" alt="image" src="https://github.com/user-attachments/assets/d74ae702-0039-4c78-b768-c8b4fc10db8e" />

**Results:**
- Train Accuracy: ~0.95  
- Test Accuracy: ~0.815  

### Convolutional Neural Network (CNN)

Trained directly on raw 28×28 images.

<img width="567" height="463" alt="image" src="https://github.com/user-attachments/assets/65016332-e2e4-4703-9d3d-3da23873b044" />

**Results:**
- Test Accuracy: ~0.9175  

## Model Comparison

<img width="607" height="458" alt="image" src="https://github.com/user-attachments/assets/8a5fea04-c0a5-4a5a-9c80-72461233821f" />

| Model          | Accuracy | Notes |
|----------------|--------|------|
| CNN            | ~0.92  | Best performance, learns spatial features |
| Random Forest  | ~0.90  | Strong performance on engineered features |
| Custom SVM     | ~0.82  | Interpretable but less flexible |

## Key Insights

- CNN performs best due to spatial feature learning  
- DCT-based features still achieve strong results with reduced dimensionality  
- Tradeoff:
  - Feature engineering → interpretable, efficient  
  - Deep learning → higher accuracy, less interpretable  

## Tech

- Python  
- NumPy / SciPy  
- Scikit-learn  
- PyTorch  
- Matplotlib  

## How to Run

```bash
pip install numpy scipy scikit-learn torch matplotlib
