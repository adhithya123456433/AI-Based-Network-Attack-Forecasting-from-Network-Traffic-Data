# AI-Based-Network-Attack-Forecasting-from-Network-Traffic-Data

## Team Members

| Register Number | Name |
|---|---|
| 24BAD004 | Adhithya K |
| 24BAD007 | Akhil C |
| 24BAD014 | Aswin S |
| 24BAD048 | Jeevaprakash S |
| 24BAD051 | Kabileeshwaran J |

## 1. Introduction

Network attacks can cause data loss and security problems. Traditional security systems usually detect attacks after suspicious activities begin.

This project uses deep learning to analyse network traffic and predict possible future attacks. It also uses Explainable AI to identify the important features behind predictions. A web dashboard displays predictions and security alerts.

## 2. Objectives

- Analyse network traffic using deep learning.
- Predict possible attacks up to 20 seconds ahead.
- Identify different attack stages.
- Explain predictions using Explainable AI.
- Display predictions and alerts on a web dashboard.
- Compare the model with traditional machine learning models.

## 3. System Architecture

```text
      Network Traffic Data
               |
               v
          Flow Encoder
               |
               v
        GRU Forecast Model
               |
               v
       Future Attack Prediction
               |
               v
         Explainable AI
               |
               v
          Web Dashboard
               |
               v
          Security Alerts
```

## 4. Technologies Used

- Python
- PyTorch
- NumPy and Pandas
- Scikit-learn
- FastAPI and Uvicorn
- HTML, CSS and JavaScript
- Chart.js and Matplotlib

## 5. Dataset

The system can use synthetic network traffic data and public datasets such as CIC-IDS2017.

The data may contain packet counts, bytes transferred, TCP flags, destination ports, and connection statistics.

The dataset is processed before being used to train and test the model.

## 6. Main Features

- Network traffic analysis
- Future attack prediction
- Attack stage classification
- Explainable AI using Integrated Gradients
- CSV file upload for analysis
- Attack scenario simulation
- Web dashboard with security alerts
- Comparison with baseline models

## 7. Installation

### Step 1: Download the Project

Download the project folder and open it in VS Code.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 8. How to Run

Run these commands from the project folder.

**Generate sample data:**

```bash
python generate_sample_data.py
```

**Train the model:**

```bash
python train_model.py
```

**Test attack predictions:**

```bash
python test_prediction.py
```

**Evaluate the model:**

```bash
python evaluate_model.py
```

**Compare models:**

```bash
python compare_models.py
```

**Start the dashboard:**

```bash
python run_dashboard.py
```

Open the local URL shown in the terminal to access the dashboard.

These commands require the corresponding scripts and dependencies to be implemented correctly.

## 9. Model Evaluation

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Mean Squared Error

The prediction horizons are 5, 10, 15, and 20 seconds. Actual performance must be measured through testing.

## 10. Conclusion

This project aims to predict possible network attacks using deep learning. It combines network traffic analysis, future attack prediction, and Explainable AI.

The dashboard helps users understand predictions and view security alerts. Testing on real-world datasets will help determine the effectiveness of the proposed system.

## 11. References

1. Ha, D., and Schmidhuber, J. (2018). *World Models*.
2. Sundararajan, M., Taly, A., and Yan, Q. (2017). *Axiomatic Attribution for Deep Networks*.
3. Hutchins, E. M., Cloppert, M. J., and Amin, R. M. (2011). *Intelligence-Driven Computer Network Defense Informed by Analysis of Adversary Campaigns and Cyber Kill Chains*.
4. Sharafaldin, I., Lashkari, A. H., and Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*.
