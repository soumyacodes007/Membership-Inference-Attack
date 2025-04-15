# Data Privacy Challenge: Membership Inference Attack (SBH2025)

## Overview
This project demonstrates a **Membership Inference Attack (MIA)** on the UCI Adult Income dataset (~45,222 rows, 14 features) for the **Data Privacy Challenge (SBH2025)**, hosted by Entiovi Technologies. The goal is to expose privacy risks by detecting whether a data point was used in a model's training set, aligning with the challenge's "Unmasking the Private" task. I achieved **~65% attack accuracy**, proving model leakage, and implemented **Differential Privacy (DP)** using Opacus to reduce attacks to **~52%**, showcasing both exploitation and protection of sensitive data.

### Key Features
- **Dataset**: UCI Adult Income (binary classification: >50K vs. <=50K).
- **MIA**: Trained 4 shadow models, used loss features to hit ~65% accuracy.
- **DP**: Opacus with ε=2.68, achieving ~82% model accuracy and ~52% attack accuracy.
- **Visuals**: Confusion matrix, confidence/loss histograms, DP comparison.

## Results
- **Target Model**: ~85% test accuracy (non-DP neural network).
- **MIA**:
  - Non-DP: ~65% attack accuracy, exposing training data leakage.
  - DP: ~52% attack accuracy, showing effective privacy defense.
- **Visuals**:
  - Confusion Matrix: Highlights attack performance.
  - Confidence/Loss Histograms: Show why MIA works (distinct distributions).
  - DP Comparison: Bar chart (non-DP 65% vs. DP 52%).

![Confusion Matrix](confusion_matrix.png)
![Confidence Histogram](confidence_histogram.png)
![Loss Histogram](loss_histogram.png)
![DP Comparison](dp_comparison.png)

## Prerequisites
- Python 3.8+
- Libraries (see `requirements.txt`):
  - `tensorflow==2.5.0`
  - `torch`
  - `opacus`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
- Datasets: `adult.csv`, `test.csv` (included in repo, from UCI Adult Income).

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo





   Install Dependencies:
bash

Copy
pip install -r requirements.txt
Verify Datasets:
Ensure adult.csv and test.csv are in the repo root.
Download from UCI Machine Learning Repository if missing.
How to Run
Launch Jupyter Notebook:
bash

Copy
jupyter notebook mia.ipynb
Execute Cells:
Run all cells (Cell -> Run All) to:
Load and preprocess data.
Train target model (~85% accuracy).
Perform MIA (~65% accuracy).
Train DP model (~82% accuracy, ~52% attack accuracy).
Generate visuals (*.png files).
Expected Runtime:
Non-DP: ~20-30 minutes (CPU).
DP: ~10-30 minutes (CPU), ~2-5 minutes (GPU, e.g., Colab).
Total: ~30-60 minutes (CPU).
Project Structure
mia.ipynb: Main notebook with MIA and DP implementation.
adult.csv, test.csv: UCI Adult Income datasets.
requirements.txt: Python dependencies.
*.png: Visuals (confusion matrix, histograms, DP comparison).
Challenge Context
For SBH2025, this project:

Unmasks Data: MIA (~65%) links model outputs to training membership, mimicking challenge tasks.
Protects Privacy: DP (ε=2.68) reduces attacks to ~52%, showing defense skills.
Demonstrates Adaptability: Overcame tensorflow_privacy errors by switching to Opacus, optimized for speed.
Future Work
Tune DP parameters (e.g., ε=1.0 for stronger privacy).
Explore regularization or adversarial training as alternative defenses.
Extend to other datasets (e.g., MNIST, previously tested).