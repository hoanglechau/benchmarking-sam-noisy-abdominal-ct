# Benchmarking SAM & MedSAM Robustness under Noisy Abdominal CT Conditions

**A Benchmark Study Evaluating Segment Anything Model Performance on Corrupted Abdominal CT Images**

**Author**: Hoang Le Chau

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains a comprehensive benchmark study evaluating the robustness of Segment Anything Model (SAM) and its medical domain adaptation (MedSAM) under realistic noisy medical imaging conditions. The project systematically assesses model performance degradation when medical images are corrupted by various noise types and artifacts commonly encountered in clinical CT acquisition.

### Research Objectives

1. **Robustness Evaluation**: Assess SAM-based model performance under various noise and artifact types
2. **Noise Sensitivity Analysis**: Quantify segmentation degradation across increasing noise intensities
3. **Model Comparison**: Benchmark SAM vs. MedSAM on corrupted medical images
4. **Failure Mode Identification**: Provide qualitative visualizations revealing boundary instability and confidence degradation
5. **Scientific Dissemination**: Produce publication-ready analysis for CVPR workshop submission

## 🎯 Key Contributions

- **Systematic Noise Injection Pipeline**: 6 noise types × 3 intensity levels (mild, moderate, severe)
- **Comprehensive Evaluation Framework**: 5 segmentation metrics computed across 3,800+ predictions
- **Multi-Dataset Validation**: Evaluation on liver and spleen CT segmentation tasks
- **Publication-Ready Visualizations**: Performance degradation curves, heatmaps, and failure case analysis
- **Reproducible Research**: Complete end-to-end pipeline from data preparation to analysis

## 📊 Datasets

This study uses the [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/) datasets:

- **Task 03 - Liver**: 131 training samples, CT modality
- **Task 09 - Spleen**: 41 training samples, CT modality

Both datasets consist of 3D abdominal CT volumes with expert annotations. We extract 2D axial slices for controlled 2D segmentation analysis.

## 🔬 Methodology

### 1. Noise Injection & Artifact Simulation

Six types of synthetic perturbations simulating realistic clinical artifacts:

| Noise Type | Description | Intensity Levels |
|------------|-------------|------------------|
| **Gaussian Noise** | Additive white noise | Mild, Moderate, Severe |
| **Poisson Noise** | Signal-dependent noise | Mild, Moderate, Severe |
| **Salt & Pepper** | Random impulse noise | Mild, Moderate, Severe |
| **Motion Blur** | Patient movement artifacts | Mild, Moderate, Severe |
| **Intensity Inhomogeneity** | Bias field corruption | Mild, Moderate, Severe |
| **Low Contrast** | Reduced image contrast | Mild, Moderate, Severe |

**Total Variants**: 18 noise configurations + 1 clean baseline = 19 conditions per dataset

### 2. Models Evaluated

- **SAM (Segment Anything Model)**: Original foundation model
- **MedSAM**: Medical domain-adapted variant

Both models evaluated using automatic segmentation mode with consistent inference settings.

### 3. Evaluation Metrics

Comprehensive quantitative assessment using:

- **Dice Coefficient**: Overlap-based similarity measure
- **IoU (Jaccard Index)**: Intersection over union
- **Precision & Recall**: Classification performance
- **Hausdorff Distance**: Boundary accuracy metric
- **Stability Metrics**: Performance variance across noise levels

## 📁 Repository Structure

```
├── Notebook_01_Data_Exploration.ipynb       # Dataset loading and 2D slice extraction
├── Notebook_02_Noise_Injection.ipynb        # Noise generation pipeline (6 types × 3 levels)
├── Notebook_03_Model_Inference_Clean.ipynb  # Baseline performance on clean data
├── Notebook_04_Model_Inference_Noisy.ipynb  # Inference on all 18 noise variants
├── Notebook_05_Evaluation_Metrics.ipynb     # Comprehensive metric computation
├── Notebook_06_Visualization.ipynb          # Publication-ready figures and analysis
└── README.md                                # This file
```

### Notebook Workflow

1. **Notebook 01**: Data exploration, volume inspection, 2D slice extraction (50 slices per organ)
2. **Notebook 02**: Systematic noise injection creating 900 corrupted images per dataset
3. **Notebook 03**: Model setup and clean data inference establishing baseline performance
4. **Notebook 04**: Batch inference on all noise variants (76 total prediction sets)
5. **Notebook 05**: Metric computation across 3,800 predictions
6. **Notebook 06**: Statistical analysis, visualization, and failure mode investigation

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA-enabled GPU (recommended)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/hoanglechau/benchmarking-sam-noisy-abdominal-ct.git
cd benchmarking-sam-noisy-abdominal-ct

# Install dependencies
pip install torch torchvision
pip install segment-anything
pip install nibabel numpy pandas matplotlib seaborn scipy scikit-image opencv-python tqdm
```

### Dataset Setup

1. Download Medical Segmentation Decathlon datasets:
   - [Task 03 - Liver](http://medicaldecathlon.com/)
   - [Task 09 - Spleen](http://medicaldecathlon.com/)

2. Update data paths in notebooks to point to your downloaded datasets

### Running the Pipeline

Execute notebooks sequentially (01 → 06) to reproduce the complete analysis:

```bash
jupyter notebook Notebook_01_Data_Exploration.ipynb
# ... continue through all notebooks
```

## 📈 Key Results

### Performance Degradation Summary

**Liver Dataset:**
- Clean baseline performance maintained across both models
- Severe noise causes dramatic segmentation failure (Dice < 0.10)
- Salt & pepper noise most harmful (Dice drop: -0.037)

**Spleen Dataset:**
- Models show poor baseline performance (Dice ~0.04-0.10)
- Noise exacerbates segmentation challenges
- Salt & pepper noise most harmful (Dice drop: -0.038)

### Noise Impact Ranking (Most to Least Harmful)

**Liver**: Salt & Pepper > Gaussian > Low Contrast > Intensity Inhomogeneity > Poisson > Motion Blur

**Spleen**: Salt & Pepper > Gaussian > Intensity Inhomogeneity > Poisson > Motion Blur > Low Contrast

### Model Comparison

- **SAM**: Higher baseline performance but minimal robustness to noise
- **MedSAM**: Shows complete segmentation failure across most noise conditions, suggesting domain adaptation does not confer noise robustness

## 📊 Visualizations

The study includes comprehensive visualizations:

- **Performance degradation curves**: Metric trends across noise intensities
- **Comparison heatmaps**: Cross-model and cross-noise performance matrices
- **Failure case galleries**: Qualitative examples of segmentation breakdown
- **Statistical significance tests**: Rigorous comparison between conditions

## 🔍 Key Insights

1. **Baseline Limitations**: Out-of-the-box SAM/MedSAM show poor performance on medical CT even without noise
2. **Severe Degradation**: All noise types cause significant performance drops, with Dice coefficients falling below 0.10
3. **Texture Sensitivity**: Salt & pepper and Gaussian noise most disruptive to segmentation
4. **Domain Gap Persists**: Medical adaptation (MedSAM) does not provide noise robustness
5. **Clinical Implications**: Current foundation models require substantial enhancement for clinical deployment

## 🛠️ Technical Details

- **Compute Environment**: Google Colab Pro+ with NVIDIA A100 GPU
- **Inference Time**: ~5 seconds per image per model
- **Total Computation**: ~157 minutes for liver dataset, ~157 minutes for spleen dataset

## 🔮 Future Directions

**Phase 2 Extensions** (Months 4-6):

1. Expand to 4-5 datasets across multiple modalities (X-ray, MRI, ultrasound)
2. Include additional SAM variants (SAM2, domain-specific adaptations)
3. Compare against conventional segmentation models (U-Net, nnU-Net)
4. Implement adapter-based robustness enhancement
5. Add uncertainty quantification analysis
6. Conduct clinical relevance assessment

## 📧 Contact

**Author**: Hoang Le Chau
**Email**: hallo.hoanglechau@gmail.com