# SAT5114-Spring-2026_hasham_Automated-Pneumonia-Detection-in-Chest-X-ray-Images-Using-Deep-Learning
# Automated Pneumonia Detection in Chest X-ray Images

**AI in Health Research (SAT5114)** course.This project implements and compares Deep Learning architectures to automate the detection of pneumonia in pediatric chest X-ray images.

## 📌 Project Overview
Pneumonia is a leading cause of global mortality, especially in resource-limited settings where radiologist shortages lead to critical diagnostic delays.  This project establishes technical benchmarks for clinical deployment by classifying images into 'Normal' and 'Pneumonia' categories using state-of-the-art Convolutional Neural Networks (CNNs). 

## 📊 Dataset
[cite_start]The study utilizes the **Kaggle Chest X-Ray Images (Pneumonia)** dataset, curated from 5,860 pediatric patients (ages 1-5). 
- **Normal cases:** 1,583 (27%) 
- **Pneumonia cases:** 4,273 (73%)
- **Expert Validation:** All images were expert-labeled by clinicians at Guangzhou Women and Children's Medical Center. 

## 🛠️ Methodology & Tech Stack
- **Models:** Compared **ResNet-50** and **DenseNet-121** architectures. 
- **Approach:** Applied **Transfer Learning** (ImageNet pre-trained weights) with selective fine-tuning. 
- **Optimization:** Addressed class imbalance using weighted loss (1.5:0.8) and medical-appropriate data augmentation. 
- **Interpretability:** Integrated **Grad-CAM** visualizations to highlight clinical features like lung infiltrates and consolidation. 

## 📈 Key Results 
| Model | Accuracy | Sensitivity | AUC-ROC | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | 88.3% | 87.4% | 0.946 | 23.85M |
| **DenseNet-121** | 89.7% | 97.7% | 0.955 | 7.17M |

*Note: **DenseNet-121** proved most effective for resource-constrained settings due to its superior parameter efficiency (70% fewer parameters than ResNet-50).

## 📂 Repository Structure
- `/src`: Python source code for model training and evaluation.
- `/docs`: Project PDF report and PPT presentation.
- `/data`: Links and scripts to access the Kaggle dataset.

## 👥 Authors
- Hasham Khan
- Moses Nihongo
Automated Pneumonia Detection in Chest X-ray Images Using Deep Learning
