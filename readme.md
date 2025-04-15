# Membership Inference Attack & Differential Privacy Study

## Project Overview
This project demonstrates the implementation of a Membership Inference Attack (MIA) on the UCI Adult Income dataset, followed by privacy protection using Differential Privacy (DP). The work explores both the vulnerability and protection aspects of machine learning models, achieving:
- 62.76% MIA accuracy on non-DP model
- 83.16% model accuracy with DP protection
- ε=2.68 privacy budget (δ=1e-5)

## Dataset
- **Source**: UCI Adult Income dataset
- **Size**: ~45,222 rows, 14 features
- **Task**: Binary classification (income >50K vs ≤50K)
- **Split**: 60% training (~27,132 samples), 20% validation, 20% test

## Methodology

### Data Preprocessing
- Handled missing values (replaced '?' with NaN, dropped incomplete rows)
- Standardized income labels
- Applied LabelEncoder for categorical features
- Used StandardScaler for numerical features

### Target Model Architecture
```python
Sequential:
- Dense(128, ReLU)
- Dense(64, ReLU)
- Dense(32, ReLU)
- Dense(1, sigmoid)
```
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Training: 100 epochs
- Performance: ~85% test accuracy

### Membership Inference Attack
- Implemented 4 shadow models
- Features: Class probabilities and per-sample loss
- Attack Model: LogisticRegression
- Results:
  - Accuracy: 62.76%
  - Precision: 75.19%
  - Recall: 75.14%

### Differential Privacy Implementation
- Framework: Opacus (PyTorch)
- Configuration:
  - Target ε: 3.0
  - Achieved ε: 2.68
  - δ: 1e-5
  - Batch size: 256
  - Epochs: 20
  - Max gradient norm: 1.0
- Results:
  - Model Accuracy: 83.16%
  - Attack Accuracy: 66.10%

## Results & Visualizations
The project includes several visualizations:
- Confusion Matrix for attack performance
- Confidence distribution histograms
- Loss distribution histograms
- DP vs Non-DP comparison charts

## Key Findings
1. Successfully demonstrated model vulnerability through MIA
2. Achieved practical privacy-utility trade-off with DP
3. Identified areas for improvement in DP implementation

## Prerequisites
```
Python 3.8+
tensorflow==2.5.0
torch
opacus
numpy
pandas
scikit-learn
matplotlib
seaborn
```

## Installation & Usage
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure dataset files (`adult.csv`, `test.csv`) are in the project directory
4. Run the main scripts for:
   - Target model training
   - MIA implementation
   - DP model training

## Future Work
- Explore stricter privacy budgets (ε=1.0)
- Implement additional regularization techniques
- Test adversarial training approaches
- Optimize DP implementation for better privacy-utility balance

## License
[Insert License Information]

## Acknowledgments
- UCI Machine Learning Repository for the Adult Income dataset
- Opacus team for the DP implementation framework
