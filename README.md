**DNS Tunneling Detection with Multilabel Classification**

This project implements a machine learning and deep learning pipeline for detecting DNS tunneling attacks using the UNSW-NB15 dataset. The model applies feature selection, preprocessing, normalization, and multilabel classification techniques to distinguish between normal and malicious traffic.

**Project Overview**

DNS tunneling is a technique used by attackers to exfiltrate data or establish command-and-control channels over the DNS protocol. Traditional firewalls and intrusion detection systems often fail to detect such traffic due to its similarity with legitimate DNS requests.
This project leverages scikit-learn and TensorFlow to build and evaluate machine learning models that can effectively classify DNS tunneling traffic.

**Key Features:**

- Data preprocessing (handling nulls, encoding categorical features, normalization).
- Recursive Feature Elimination (RFE) for feature selection.
- Random Forest classifier for feature importance analysis.
- Deep Neural Networks (Keras Sequential model) for multilabel classification.
- Training, validation, and early stopping with custom callbacks.
- Evaluation using precision, recall, F1-score, and ROC-AUC.

 **Dataset**

The models are trained and evaluated using the UNSW-NB15 dataset, which is widely used in cybersecurity research.

Files used:

- UNSW-NB15_3.csv: main dataset
N- USW-NB15_features.csv: feature descriptions

Target variables:

- Label: indicates normal (0) or attack (1) traffic.
- service: categorical service label (used as part of multilabel encoding).

**Workflow**
1. Data Preprocessing

- Missing values removed.
- Encoded categorical features (proto, state, service).
- Dropped irrelevant fields (srcip, dstip, etc.).
- Normalized features using MinMaxScaler.

2. Feature Selection

- Applied Recursive Feature Elimination (RFE) with RandomForest.
- Selected top 10 features, including sbytes, dbytes, sttl, dttl, tcprtt, etc.

3. Modeling

- Random Forest Classifier for baseline evaluation.
- Deep Neural Network (DNN) with multiple dense layers and sigmoid activation for multilabel outputs.
- MLP model tested as an alternative architecture.

4. Training Strategy

- Train/test split (60/40).
- Early stopping with a custom callback to prevent overfitting.
- Training accuracy reached 99.4%, test accuracy reached 99.4%.

5. Evaluation Metrics

- Classification report (Precision, Recall, F1-score).
- ROC curve and AUC analysis.
- Weighted F1 score: 0.98.
- High precision (0.96) and recall (0.99).

**Results**

Random Forest baseline: ~98% macro F1-score.

DNN model:

Training accuracy: 99.4%
Test accuracy: 99.4%
Weighted F1-score: 0.98
ROC-AUC: ~0.99

The results demonstrate that the proposed model is highly effective in detecting DNS tunneling traffic with minimal false positives.

**Installation & Usage
Requirements
**
Python 3.8+

TensorFlow 2.x

Scikit-learn

Pandas

NumPy

Matplotlib


**Future Improvements**

Experiment with other deep learning architectures (CNN, LSTM for sequential data).

Apply class balancing techniques (SMOTE, oversampling).

Explore unsupervised anomaly detection for zero-day attacks.

Deploy as a real-time intrusion detection module
