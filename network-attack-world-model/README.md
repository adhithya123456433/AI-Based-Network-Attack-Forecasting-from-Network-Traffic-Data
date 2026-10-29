# AI-Based-Network-Attack-Forecasting-from-Network-Traffic-Data

## Team Members

| Register Number | Name |
|---|---|
| 24BAD004 | Adhithya K |
| 24BAD007 | Akhil C |
| 24BAD014 | Aswin S |
| 24BAD048 | Jeevaprakash S |
| 24BAD051 | Kabileeshwaran J |
---

## Project Overview

Traditional Network Intrusion Detection Systems (NIDS) operate reactively: they flag malicious activity only after suspicious packets have entered the network or an exploit has already succeeded. By the time security teams receive an alert, systems may already be breached and sensitive data compromised.

NetForecaster introduces a proactive defense framework powered by a Deep Learning Latent Dynamics World Model. Rather than analyzing static, isolated snapshots of network packets, our system continuously models the underlying temporal behavior of enterprise network traffic over 30-second observation windows and forecasts network states 5 steps into the future.

By projecting network telemetry forward in time, the system identifies the early preparatory phases of cyber attacks (such as port scanning and brute-force authentication attempts) and issues actionable warnings before the attacker reaches the exploitation or data theft stages.

## 1. Project Architecture 

```
Past Observation Window X[t-W : t]
                 │
                 ▼
     ┌───────────────────────┐
     │  Temporal 1D-CNN      │  (LayerNorm + Dropout = 0.25)
     │  Observation Encoder  │  Compresses high-dimensional flows to latent space
     └───────────────────────┘
                 │
                 ▼ Latent state z[t]
     ┌───────────────────────┐
     │  Residual GRU Core    │  Learns transition physics:
     │  (World Model)        │  z[t+1] = f_theta(z[t])
     └───────────────────────┘
                 │
                 ▼ Latent trajectory z[t+1:t+k]
        ┌────────┴────────────────────────┐
        ▼                                 ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Future State Decoder  │       │ Attack Stage Head     │
│ Forecasts telemetry   │       │ Categorizes Kill-Chain│
│ X_hat[t+1 : t+k]      │       │ (Normal, Recon, etc.) │
│ (Huber Loss)          │       │ (Focal Loss)          │
└───────────────────────┘       └───────────────────────┘
        │                                 │
        └───────────────┬─────────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Integrated Gradients Engine  │
        │  Explains top feature triggers│
        └───────────────────────────────┘
```

---

## 2. Overfitting & Underfitting Prevention Strategy

- **Zero Temporal Data Leakage**: Enforces strict chronological forward-chaining splits (`70% train`, `15% validation`, `15% test`). Scalers are fitted strictly on the training segment.
- **Latent Bottleneck**: Constrains latent state dimension ($d_z = 64$) to prevent memorization of noise.
- **Regularization Stack**: Spatial and recurrent dropout ($p=0.25$), $L_2$ weight decay ($10^{-4}$), and Layer Normalization.
- **Residual Skip Connections**: Connects recurrent blocks to preserve gradient flow and prevent underfitting.
- **Class Imbalance Mitigation**: Uses multi-class **Focal Loss** ($\gamma=2.0$) with class weighting, preventing the optimizer from trivially defaulting to the majority "Normal" class.
- **Early Stopping**: Monitored on Validation Macro-F1 score with automatic best checkpoint restoration.

---

## 3. Directory Structure

```
network-attack-world-model/
├── data_pipeline.py      # Chronological split, normalization, & sliding window tensors
├── losses.py             # Multi-class Focal Loss & Huber Joint Multi-Task Objective
├── models.py             # World Model (1D-CNN + Residual GRU) and Baselines (MLP, LSTM)
├── explainability.py     # PyTorch Autograd Integrated Gradients Feature Attribution
├── train_and_evaluate.py # Training loop, early stopping, and metric evaluation
├── main.py               # End-to-end execution and benchmark comparison
├── app.py                # FastAPI server & inference backend
├── static/               # Interactive web platform frontend
│   └── index.html        # 5-Page Dashboard (Overview, Traffic, Forecast, Alerts, Evaluation)
├── requirements.txt      # Project Python dependencies
└── run_web_app.bat       # Single-click Windows web application launcher
```

---

## 4. Installation & Execution

### Setup Environment
```bash
git clone https://github.com/your-username/network-attack-world-model.git
cd network-attack-world-model
pip install -r requirements.txt
```

### Train & Evaluate Model
```bash
python main.py
```

### Launch Interactive Web Defense Dashboard
```bash
python app.py
```
Or double-click `run_web_app.bat` to launch the server on `http://127.0.0.1:8080`.
