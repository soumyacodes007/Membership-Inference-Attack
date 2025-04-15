# Membership Inference Attack & Differential Privacy Study



TLDR : This project implements a Membership Inference Attack (MIA) on the UCI Adult Income dataset to
expose privacy vulnerabilities in machine learning models. MIAs determine whether a data point was
part of a model’s training set, aligning with the challenge’s goal of unmasking protected data. Using a
neural network classifier, I achieved a non-DP MIA accuracy of 62.76%, proving significant leakage. To
counter this, I applied Differential Privacy (DP) with Opacus, reducing attack accuracy to 66.10%
while maintaining 83.16% model accuracy. This report details the methodology, results, and
implications, showcasing my ability to exploit and mitigate privacy risks.


for detailed documentation visist this = https://drive.google.com/file/d/1dRymDBeQ3jROaAAgfvVOhAhWb7jIotw6/view?usp=sharing


## Dataset
- **Source**: UCI Adult Income dataset
- **Size**: ~45,222 rows, 14 features
- **Task**: Binary classification (income >50K vs ≤50K)
- **Split**: 60% training (~27,132 samples), 20% validation, 20% test


Dataset = https://archive.ics.uci.edu/dataset/2/adult

## Project Overview
This project demonstrates the implementation of a Membership Inference Attack (MIA) on the UCI Adult Income dataset, followed by privacy protection using Differential Privacy (DP). The work explores both the vulnerability and protection aspects of machine learning models, achieving:
- 62.76% MIA accuracy on non-DP model
- 83.16% model accuracy with DP protection
- ε=2.68 privacy budget (δ=1e-5)

Target Model

A TensorFlow neural network (Sequential: 128-64-32-1 layers, ReLU, sigmoid output) was trained for
binary classification using Adam optimizer and binary cross-entropy loss. It ran for 100 epochs to
induce overfitting, achieving ~85% test accuracy, setting the stage for MIA exploitation.

Membership Inference Attack

To perform the MIA, I trained four shadow models on random training subsets, mimicking the target
model. Each model’s outputs—probabilities for >50K and <=50K, plus per-sample loss—were
collected. The loss feature, critical for distinguishing members (training data) from non-members
(test data), was computed using binary cross-entropy. A LogisticRegression attack model was trained
on these outputs, labeling training data as members (1) and test data as non-members (0).

Differential Privacy

To mitigate MIA risks, I implemented DP using Opacus in PyTorch, training a similar neural network
(128-64-32-1) with DP-SGD. Targeting a privacy budget of ε=3.0 (δ=1e-5), I used a batch size of 256,
20 epochs, and max gradient norm of 1.0. This addressed earlier tensorflow_privacy issues,
optimizing for speed (~10-30 minutes on CPU). The DP model balanced utility and privacy, tracked via
epsilon (reaching 2.68).


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
- Framework: (PyTorch)
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
The project includes several visualizations: Opacus
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



