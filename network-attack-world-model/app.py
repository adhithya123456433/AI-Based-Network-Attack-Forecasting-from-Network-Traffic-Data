"""
app.py
------
FastAPI Backend for AI-Based Network Attack Forecasting World Model.
Serves interactive live streaming UI, processes CSV uploads,
and provides real-time model inference and Integrated Gradients explainability.
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from models import NetworkAttackWorldModel
from explainability import ExplainabilityEngine
from data_pipeline import (
    FEATURE_NAMES,
    STAGE_NAMES,
    generate_synthetic_attack_telemetry
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "world_model_weights.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

app = FastAPI(title="Network Attack Forecasting World Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Model & Scaler
device = "cpu"
model = NetworkAttackWorldModel(in_features=len(FEATURE_NAMES), latent_dim=64, horizon=5, num_classes=5, dropout=0.0)

if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"[*] Loaded trained model weights from {WEIGHTS_PATH}")
else:
    print("[!] Warning: Weights not found; initialized random model.")

model.eval()

scaler = None
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"[*] Loaded scaler from {SCALER_PATH}")

explainer = ExplainabilityEngine(model, FEATURE_NAMES, STAGE_NAMES)


class ForecastRequest(BaseModel):
    window: List[List[float]]  # [30, 10]


@app.get("/api/health")
def health():
    return {"status": "ok", "device": device, "model": "NetworkAttackWorldModel"}


@app.get("/api/sample/{scenario_type}")
def get_sample_data(scenario_type: str):
    """
    Returns a sequence of continuous network traffic for live playback and slider scrubbing.
    scenario_type: 'normal', 'recon', 'delivery', 'exploit', 'exfil', or 'full_campaign'
    """
    total_len = 350
    df = generate_synthetic_attack_telemetry(total_timesteps=total_len, random_seed=42)
    
    # Customize attack injection based on scenario request
    if scenario_type == "recon":
        df["stage"] = 0
        df.loc[60:120, "stage"] = 1
        df.loc[60:120, "syn_flag_ratio"] += 0.45
        df.loc[60:120, "rst_flag_ratio"] += 0.25
        df.loc[60:120, "dst_port_entropy"] += 0.65
        df.loc[60:120, "packet_rate"] += 90
    elif scenario_type == "delivery":
        df["stage"] = 0
        df.loc[60:120, "stage"] = 2
        df.loc[60:120, "failed_logins_rate"] += 0.60
        df.loc[60:120, "packet_size_var"] += 120
    elif scenario_type == "exploit":
        df["stage"] = 0
        df.loc[60:120, "stage"] = 3
        df.loc[60:120, "src_ip_entropy"] += 0.45
        df.loc[60:120, "flow_duration"] += 350
    elif scenario_type == "exfil":
        df["stage"] = 0
        df.loc[60:120, "stage"] = 4
        df.loc[60:120, "byte_volume"] += 120000
        df.loc[60:120, "packet_rate"] += 350
    elif scenario_type == "normal":
        df["stage"] = 0

    feature_cols = [c for c in df.columns if c != "stage"]
    raw_vals = df[feature_cols].values
    
    if scaler is not None:
        scaled_vals = scaler.transform(raw_vals)
    else:
        scaled_vals = raw_vals

    return {
        "scenario": scenario_type,
        "feature_names": feature_cols,
        "stages": df["stage"].tolist(),
        "stage_names": STAGE_NAMES,
        "scaled_data": scaled_vals.tolist(),
        "raw_data": raw_vals.tolist()
    }


# Column aliases to map common CSV headers to required feature names
COLUMN_ALIASES = {
    "packet_rate": ["packets", "packet_count", "packets_per_sec", "packetrate", "pkts", "pkt_rate"],
    "byte_volume": ["bytes", "byte_rate", "bytes_per_sec", "byte_count", "bytes_transferred", "bandwidth"],
    "syn_flag_ratio": ["syn", "syn_flags", "syn_ratio", "syn_count", "tcp_syn", "syn_rate"],
    "ack_flag_ratio": ["ack", "ack_flags", "ack_ratio", "ack_count", "tcp_ack"],
    "rst_flag_ratio": ["rst", "rst_flags", "rst_ratio", "rst_count", "tcp_rst", "resets"],
    "flow_duration": ["duration", "flow_time", "session_length", "flow_len", "connection_time"],
    "dst_port_entropy": ["port_entropy", "dst_entropy", "dest_port_entropy", "ports", "port_randomness"],
    "src_ip_entropy": ["ip_entropy", "src_entropy", "source_ip_entropy", "ip_randomness"],
    "failed_logins_rate": ["failed_logins", "failed_login_count", "login_failures", "bad_passwords", "auth_fails"],
    "packet_size_var": ["packet_var", "packet_variance", "size_var", "pkt_size_var", "size_variance"]
}


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accepts user-uploaded CSV traffic logs, validates/fills features,
    handles missing values (NaNs), non-numeric values, or different CSV formats safely.
    """
    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        except UnicodeDecodeError:
            df = pd.read_csv(io.StringIO(content.decode("latin1")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV file: {str(e)}")

    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    # Harmonize column names via aliases (case-insensitive)
    lower_col_map = {str(c).strip().lower(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in lower_col_map:
                    df[canonical] = df[lower_col_map[alias]]
                    break

    missing_cells = int(df.isna().sum().sum())
    auto_filled_cols = []
    
    # 1. Clean and harmonize required features
    for feat in FEATURE_NAMES:
        if feat in df.columns:
            # Coerce non-numeric text to numbers, fill NaNs with column median or baseline
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            median_val = df[feat].median()
            default_fill = median_val if pd.notna(median_val) else (0.2 if "entropy" in feat else (0.05 if "ratio" in feat else 100.0))
            df[feat] = df[feat].fillna(default_fill)
        else:
            auto_filled_cols.append(feat)
            # Safe domain-aware defaults for missing columns
            if "entropy" in feat:
                df[feat] = 0.20 + np.random.normal(0, 0.02, len(df))
            elif "ratio" in feat or "rate" in feat and "packet" not in feat:
                df[feat] = 0.05 + np.random.normal(0, 0.01, len(df))
            elif "duration" in feat:
                df[feat] = 250.0 + np.random.normal(0, 10, len(df))
            elif "byte" in feat:
                df[feat] = 40000.0 + np.random.normal(0, 2000, len(df))
            else:
                df[feat] = 100.0 + np.random.normal(0, 5, len(df))

    # 2. Handle files with fewer than 35 rows (ensure minimum sliding window size)
    if len(df) < 35:
        repeat_factor = int(np.ceil(35 / len(df)))
        df = pd.concat([df] * repeat_factor, ignore_index=True)

    raw_features = df[FEATURE_NAMES].values.astype(np.float32)

    # Clean any remaining infinites or NaNs
    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=1000.0, neginf=-1000.0)

    if scaler is not None:
        try:
            scaled_features = scaler.transform(raw_features)
        except Exception:
            scaled_features = raw_features
    else:
        scaled_features = raw_features

    stages = df["stage"].tolist() if "stage" in df.columns else [0] * len(df)

    # Determine plain-English health status message
    if missing_cells > 0 and len(auto_filled_cols) > 0:
        health_status = "Auto-Repaired & Adapted"
        user_message = f"Found {missing_cells} missing values and adapted {len(auto_filled_cols)} missing columns using standard baselines."
    elif missing_cells > 0:
        health_status = "Cleaned Missing Values"
        user_message = f"Successfully repaired {missing_cells} empty/missing cells with statistical medians."
    elif len(auto_filled_cols) > 0:
        health_status = "Format Adapted"
        user_message = f"Adapted format: matched {len(FEATURE_NAMES) - len(auto_filled_cols)} features and provided safe baselines for the rest."
    else:
        health_status = "100% Validated Clean Data"
        user_message = "All 10 network flow features validated with zero missing data."

    return {
        "status": "success",
        "filename": file.filename,
        "total_rows": len(df),
        "missing_cells_repaired": missing_cells,
        "health_status": health_status,
        "user_message": user_message,
        "feature_names": FEATURE_NAMES,
        "stages": stages,
        "stage_names": STAGE_NAMES,
        "scaled_data": scaled_features.tolist(),
        "raw_data": raw_features.tolist()
    }


@app.post("/api/forecast")
def run_forecast(req: ForecastRequest):
    """
    Runs model inference on the current sliding observation window [30, 10]:
    - Returns predicted future telemetry [5, 10]
    - Returns forecasted cyber kill-chain stage
    - Returns Integrated Gradients feature attribution breakdown
    - Returns friendly plain-English explanation
    """
    window_arr = np.array(req.window, dtype=np.float32)
    if window_arr.shape != (30, 10):
        raise HTTPException(status_code=400, detail=f"Expected window shape (30, 10), got {window_arr.shape}")

    tensor_in = torch.tensor(window_arr, dtype=torch.float32).unsqueeze(0)  # [1, 30, 10]

    with torch.no_grad():
        pred_future, stage_logits, _ = model(tensor_in)
        probs = torch.softmax(stage_logits, dim=-1).squeeze(0)
        pred_stage_idx = torch.argmax(probs).item()
        confidence = probs[pred_stage_idx].item()

    # Compute Feature Attribution via Integrated Gradients
    attribution_scores, _ = explainer.attribute(tensor_in, target_class=pred_stage_idx, steps=20)
    
    ranked_indices = np.argsort(attribution_scores)[::-1]
    top_features = [
        {"feature": FEATURE_NAMES[i], "attribution_pct": float(np.round(attribution_scores[i], 2))}
        for i in ranked_indices
    ]

    stage_name = STAGE_NAMES[pred_stage_idx]
    threat_level = "LOW / BENIGN" if pred_stage_idx == 0 else (
        "ELEVATED" if pred_stage_idx == 1 else (
            "HIGH" if pred_stage_idx == 2 else "CRITICAL"
        )
    )

    # Friendly human-readable explanation
    top_feat_name = top_features[0]["feature"].replace("_", " ") if top_features else "traffic volume"
    top_pct = top_features[0]["attribution_pct"] if top_features else 0.0

    if pred_stage_idx == 0:
        human_explanation = "Traffic behavior is normal and safe. Connection rates, packet sizes, and handshake flag ratios match standard enterprise patterns."
    elif pred_stage_idx == 1:
        human_explanation = f"Port scanning detected! The primary trigger was {top_feat_name} ({top_pct:.1f}% contribution). An external attacker is probing closed network ports to discover active services."
    elif pred_stage_idx == 2:
        human_explanation = f"Delivery or brute-force attack detected! The primary indicator was {top_feat_name} ({top_pct:.1f}% contribution). A burst of repeated failed logins indicates an automated credential guessing attempt."
    elif pred_stage_idx == 3:
        human_explanation = f"Exploit or system breach detected! The model identified anomalous flow duration and routing entropy ({top_feat_name} {top_pct:.1f}%), indicating unauthorized script execution."
    else:  # stage 4
        human_explanation = f"Critical data theft (exfiltration) detected! A massive surge in outbound data transfer was detected ({top_feat_name} {top_pct:.1f}%). Confidential files are being transferred out of the network."

    return {
        "stage_id": pred_stage_idx,
        "stage_name": stage_name,
        "threat_level": threat_level,
        "confidence_pct": round(confidence * 100.0, 1),
        "is_malicious": pred_stage_idx > 0,
        "predicted_future": pred_future.squeeze(0).numpy().tolist(),  # [5, 10]
        "top_features": top_features,
        "human_explanation": human_explanation
    }


# Serve sample datasets for user downloads
@app.get("/api/download/{filename}")
def download_sample_csv(filename: str):
    valid_files = {
        "clean": "sample_1_clean_normal.csv",
        "recon": "sample_2_port_scan_attack.csv",
        "missing": "sample_3_with_missing_values.csv",
        "exfil": "sample_4_data_theft_exfil.csv"
    }
    file_to_serve = valid_files.get(filename)
    if not file_to_serve:
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    filepath = os.path.join(BASE_DIR, file_to_serve)
    return FileResponse(filepath, media_type="text/csv", filename=file_to_serve)


# Serve live visual dashboard at root
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)

@app.get("/")
def serve_dashboard():
    return FileResponse(os.path.join(static_dir, "index.html"))

app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    import socket

    def find_free_port(preferred_port=8000):
        for port in [preferred_port, 8080, 8050, 8888, 5000]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) != 0:
                    return port
        return preferred_port

    port = find_free_port(8000)
    print("\n" + "="*70)
    print(f" [*] Starting World Model Web Server on http://127.0.0.1:{port}")
    print("="*70 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
