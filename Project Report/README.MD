# AI-Based Network Attack Forecasting Using a Latent Dynamics World Model

### NetForecaster – A Deep Learning Framework for Predictive Cybersecurity and Explainable Early Warning Detection

WORD DOCUMENT : https://kumaragurudtsteam-my.sharepoint.com/:w:/g/personal/adhithya_24ad_kct_ac_in/cQqC4Mf3kqXrR6li8h9R4MEaEgUC34N8JN-Wnw4DYFMUWu-SIg

## Project Overview

**NetForecaster** is a deep learning-based network attack forecasting system designed to predict potential cyber threats before they fully develop.

Traditional Network Intrusion Detection Systems (NIDS) primarily identify attacks after suspicious activities occur. NetForecaster takes a proactive approach by learning the temporal behavior of network traffic and forecasting future network states.

The system uses a **Latent Dynamics World Model** to analyze network traffic patterns, predict future telemetry values, classify cyber attack stages, and generate explainable security alerts.

Unlike conventional intrusion detection systems, NetForecaster provides an early warning capability by forecasting network behavior up to **5 steps into the future**.

## Problem Statement

Traditional network intrusion detection systems are reactive and generally generate alerts after malicious activities have already occurred.

This creates several challenges:

- Delayed identification of emerging cyber attacks.
- Limited ability to predict future network behavior.
- Difficulty understanding why an AI model identifies traffic as malicious.
- Challenges in handling noisy and incomplete network traffic data.

**Our Solution:** Develop an intelligent forecasting system that learns network traffic dynamics, predicts future attack progression, and provides explainable early warnings to network administrators.

## Key Features

- **Future Network Traffic Forecasting:** Predicts network telemetry up to 5 steps ahead.
- **Cyber Attack Stage Classification:** Identifies five network activity stages.
- **Latent Dynamics World Model:** Learns temporal network behavior using a Residual GRU architecture.
- **Explainable AI:** Uses Integrated Gradients to identify the features responsible for security alerts.
- **Interactive Dashboard:** Provides visual monitoring of network traffic and predictions.
- **CSV Upload:** Allows users to upload network traffic datasets.
- **Automatic Data Preprocessing:** Handles missing values, non-numeric values, and inconsistent column names.
- **Model Performance Analysis:** Compares the proposed model against baseline architectures.

## System Architecture

NetForecaster consists of five major processing stages.

```text
       Network Traffic Data
                |
                v
       Data Preprocessing
                |
                v
      1D-CNN Temporal Encoder
                |
                v
        Latent Representation
             (64D)
                |
                v
       Residual GRU Dynamics
                |
                v
       Future State Prediction
                |
          +-----+-----+
          |           |
          v           v
    Telemetry     Attack Stage
    Forecasting   Classification
          |           |
          +-----+-----+
                |
                v
       Integrated Gradients
                |
                v
       Explainable AI Alerts
                |
                v
       Interactive Web Dashboard
```

### 1. 1D-CNN Temporal Observation Encoder

The 1D-CNN extracts temporal patterns from network traffic sequences.

- Input window: 30 time steps.
- Input features: 10 network telemetry metrics.
- Convolution filters: 64.
- Kernel size: 3.
- Latent representation: 64 dimensions.

### 2. Residual GRU Dynamics Core

The GRU learns how network states change over time.

A residual connection helps preserve information during sequential processing.

The model autoregressively predicts future latent states up to 5 steps ahead.

### 3. Dual Output Decoders

The model contains two output branches:

**Telemetry Forecasting**

- Predicts future network traffic metrics.
- Uses Huber Loss.
- Helps identify unusual future traffic patterns.

**Cyber Attack Classification**

- Classifies network activity into five cyber attack progression stages.
- Uses class-weighted Focal Loss.
- Handles class imbalance during training.

### 4. Explainable AI – Integrated Gradients

Integrated Gradients explains why the model generates an attack alert.

It identifies the network features that contribute most to the prediction and displays their attribution scores.

This helps network administrators understand the reasons behind security warnings.

## Cyber Attack Classification

The system models five stages of network activity:

| Stage | Classification | Description |
|---|---|---|
| Stage 0 | Normal | Regular network communication |
| Stage 1 | Reconnaissance | Network scanning and port probing |
| Stage 2 | Delivery | Credential attacks and malicious payload delivery |
| Stage 3 | Exploitation | Exploiting vulnerabilities and executing malicious commands |
| Stage 4 | Exfiltration | Unauthorized extraction of sensitive data |

## Network Traffic Features

The model uses 10 network telemetry features:

| Feature | Description |
|---|---|
| packet_rate | Number of packets transmitted per second |
| byte_volume | Aggregate data volume |
| syn_flag_ratio | Proportion of TCP SYN flags |
| ack_flag_ratio | Proportion of TCP ACK flags |
| rst_flag_ratio | Proportion of TCP RST flags |
| flow_duration | Average duration of network flows |
| dst_port_entropy | Entropy of destination port numbers |
| src_ip_entropy | Entropy of source IP addresses |
| failed_logins_rate | Failed authentication attempts |
| packet_size_var | Variance in packet sizes |

## Dataset and Data Preprocessing

The project uses a domain-aware synthetic network attack telemetry generator to simulate multi-stage cyber attack scenarios.

- Total records: 4,500.
- Network features: 10.
- Attack stages: 5.
- Observation window: 30 time steps.
- Forecasting horizon: 5 steps.

### Data Preprocessing

The preprocessing pipeline includes:

- Missing value identification and median imputation.
- Non-numeric value handling.
- Intelligent column alias mapping.
- Sequence padding for shorter CSV files.
- RobustScaler normalization.

### Chronological Data Splitting

To prevent temporal data leakage, the dataset is divided chronologically.

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The RobustScaler is fitted only on the training dataset and subsequently applied to validation and testing data.

## Model Performance and Evaluation

NetForecaster is evaluated against two baseline models under the same chronological testing conditions.

| Model | Accuracy | Macro F1 | PR-AUC | Forecast MAE | Early Warning |
|---|---:|---:|---:|---:|---|
| Static MLP | 96.72% | 0.9095 | 0.9412 | N/A | 0 steps |
| Vanilla Seq2Seq LSTM | 94.07% | 0.8439 | 0.8876 | 2.0841 | 2 steps |
| **NetForecaster** | **95.16%** | **0.8782** | **0.9145** | **1.7408** | **4–5 steps** |

### Evaluation Metrics

- Classification Accuracy
- Macro F1-Score
- Precision-Recall AUC
- Mean Absolute Error (MAE)
- Confusion Matrix
- Training and Validation Loss

**Key Result:** NetForecaster achieves a 95.16% classification accuracy and provides a 4–5 step early warning capability, along with future telemetry forecasting and explainable predictions.

The Static MLP achieves higher classification accuracy but does not provide future telemetry forecasting.

## Web Application

The project includes an interactive web-based cybersecurity dashboard built using FastAPI and frontend web technologies.

### Dashboard Modules

**1. Security Overview**

- Displays network records analyzed.
- Shows benign and malicious traffic ratios.
- Displays network monitoring information.

**2. Upload & Check Traffic**

- Upload network traffic CSV files.
- Select preloaded attack scenarios.
- Automatically preprocess uploaded data.

**3. AI Future Forecast**

- Visualizes future network traffic predictions.
- Compares predicted values against actual values.
- Displays cyber attack progression.

**4. Why It Flagged – Explainable AI**

- Displays Integrated Gradients feature attribution.
- Identifies important features responsible for alerts.
- Provides understandable explanations for predictions.

**5. AI Accuracy & Testing**

- Displays model comparison results.
- Shows training and validation loss.
- Visualizes confusion matrices and evaluation metrics.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Machine Learning | Scikit-learn |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| UI Styling | Tailwind CSS |
| Data Visualization | Chart.js |
| Data Processing | NumPy, Pandas |
| Explainable AI | Integrated Gradients |
| Development Tools | VS Code, Git, GitHub |

## Installation and Execution

### Prerequisites

- Python 3.10
- Git
- Web browser

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Navigate to the Project Folder

```bash
cd AI-Based-Network-Attack-Forecasting
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

If the project contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

On Windows, execute the provided launcher:

```bash
run_web_app.bat
```

The launcher starts the FastAPI backend at:

```text
http://127.0.0.1:8080
```

It also automatically opens the default web browser.

## Project Team

**Department of Artificial Intelligence and Data Science**

**Course:** 24ADI006 – Deep Learning

**Academic Year:** 2026–2027

| S.No | Register Number | Student Name |
|---|---|---|
| 1 | 24BAD004 | Adhithya K |
| 2 | 24BAD007 | Akhil C |
| 3 | 24BAD014 | Aswin S |
| 4 | 24BAD048 | Jeevaprakash S |
| 5 | 24BAD051 | Kabileeshwaran J |

## Future Enhancements

- Integration of Multi-Head Self-Attention mechanisms.
- GPU-based deployment using TensorRT.
- High-throughput network traffic processing.
- Integration with Software-Defined Networking (SDN) controllers.
- Automated firewall rule deployment based on early warning alerts.

## References

1. Goodfellow, I., Bengio, Y., and Courville, A. *Deep Learning*. MIT Press, 2016.
2. Chollet, F. *Deep Learning with Python*, 2nd Edition, 2021.
3. Vaswani et al. "Attention Is All You Need." NeurIPS, 2017.
4. Sundararajan, M., Taly, A., and Yan, Q. "Axiomatic Attribution for Deep Networks." ICML, 2017.
5. Lin et al. "Focal Loss for Dense Object Detection." 2017.
6. Ha, D. and Schmidhuber, J. "Recurrent World Models Facilitate Policy Evolution." NeurIPS, 2018.

---

## Conclusion

NetForecaster demonstrates how deep learning and latent dynamics modeling can be applied to proactive cybersecurity.

By combining 1D-CNN feature extraction, Residual GRU-based temporal modeling, multi-task learning, and Integrated Gradients, the system goes beyond traditional reactive intrusion detection.

It provides future network traffic forecasting, cyber attack stage classification, and explainable early warning alerts through an interactive web platform.

**NetForecaster – Predicting Cyber Threats Before They Escalate.**
