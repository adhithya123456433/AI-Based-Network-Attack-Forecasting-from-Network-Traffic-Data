"""
Custom SOC Cybersecurity Web Dashboard Backend (FastAPI).
Exposes REST APIs for:
- Live streaming network flow scenarios
- World Model latent dynamics & multi-step forecasting
- Integrated Gradients Explainable AI (XAI) feature attributions
- Custom CSV file uploads & analysis
"""
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import pandas as pd
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.world_model import NetworkWorldModel
from data.killchain_simulator import NetworkKillChainSimulator
from data.real_data_adapter import RealNetworkDataAdapter
from explainability.attribution import IntegratedGradientsExplainer

FRIENDLY_FEATURE_NAMES = {
    "failed_conn_ratio": "Repeated Failed Login Attempts",
    "dst_port_entropy": "Testing Many Network Doors (Port Scan)",
    "syn_flag_ratio": "Sudden Rush of Connection Requests",
    "flow_bytes_per_sec": "High Data Transfer Rate",
    "byte_ratio_fwd_bwd": "Large Amount of Outgoing Data",
    "conn_density_window": "Too Many Connections from One Device",
    "payload_entropy": "Hidden or Encrypted Data Packages",
    "fwd_pkts_per_sec": "Rapid Speed of Outgoing Messages",
    "rst_flag_ratio": "Connections Being Rejected",
    "flow_duration_ms": "Unusually Long Connection",
    "fwd_pkt_len_mean": "Large Message Size Sent",
    "bwd_pkt_len_mean": "Large Message Size Received",
    "ack_flag_ratio": "Normal Acknowledged Traffic",
    "fwd_iat_mean_ms": "Time Gap Between Sent Messages",
    "bwd_iat_mean_ms": "Time Gap Between Replies",
    "bwd_pkts_per_sec": "Incoming Messages Speed"
}

FRIENDLY_STAGE_NAMES = [
    "Normal & Safe",
    "Looking for Weaknesses (Scanning)",
    "Break-In Attempt (Password / Exploit Attack)",
    "Spreading Infection to Other Computers",
    "Data Theft or System Crash"
]

SIMPLE_RECOMMENDATIONS = {
    0: "Everything is safe and healthy. Standard normal network traffic.",
    1: "Someone is scanning your network searching for weak doors. Recommended Action: Block the suspicious IP address and turn on firewall rate-limiting.",
    2: "A hacker is actively trying to guess passwords or break in. Recommended Action: Temporarily lock the targeted account and alert administrators.",
    3: "An infected device is trying to jump to other computers on your network. Recommended Action: Disconnect the compromised computer from the network immediately.",
    4: "Data is actively being copied out or the server is being flooded. Recommended Action: Shut off outbound internet access for this server to prevent data loss."
}

app = FastAPI(title="AI-Based Network Attack Forecasting World Model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
DEVICE = "cpu"
ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_world_model.pt")
if not os.path.exists(ckpt_path):
    raise RuntimeError(f"Checkpoint not found at {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=DEVICE)
model = NetworkWorldModel(
    num_features=ckpt["config"]["num_features"],
    latent_dim=ckpt["config"]["latent_dim"],
    hidden_dim=ckpt["config"]["hidden_dim"],
    num_stages=ckpt["config"]["num_stages"],
    forecast_horizon=ckpt["config"]["forecast_horizon"]
).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

explainer = IntegratedGradientsExplainer(model, steps=20)
adapter = RealNetworkDataAdapter()

# In-memory active scenario buffer
ACTIVE_STREAM = {
    "X": None,
    "Y": None,
    "name": "Default Scenario"
}


def load_default_scenario(scenario_type="full_attack", seed=42):
    sim = NetworkKillChainSimulator(random_seed=seed)
    X, Y = sim.generate_episode(episode_type=scenario_type, length=120)
    ACTIVE_STREAM["X"] = X
    ACTIVE_STREAM["Y"] = Y
    ACTIVE_STREAM["name"] = f"Simulated {scenario_type} (Seed {seed})"


load_default_scenario()


@app.get("/api/info")
def get_info():
    return {
        "status": "ARMED",
        "model": "Latent State Space World Model (RSSM-GRU)",
        "features": config.FEATURE_NAMES,
        "stages": config.STAGE_NAMES,
        "window_size": config.WINDOW_SIZE,
        "forecast_horizon": config.FORECAST_HORIZON,
        "time_step_seconds": config.TIME_STEP_SECONDS,
        "lead_time_seconds": config.FORECAST_HORIZON * config.TIME_STEP_SECONDS,
        "stream_length": len(ACTIVE_STREAM["X"]),
        "stream_name": ACTIVE_STREAM["name"]
    }


@app.get("/api/load_scenario")
def load_scenario(scenario_type: str = "full_attack", seed: int = 42):
    load_default_scenario(scenario_type, seed)
    return {
        "message": f"Loaded scenario {scenario_type}",
        "length": len(ACTIVE_STREAM["X"]),
        "name": ACTIVE_STREAM["name"]
    }


@app.get("/api/load_sample_csv")
def load_sample_csv():
    load_default_scenario("full_attack", 42)
    return {
        "message": "Reset to default network stream",
        "length": len(ACTIVE_STREAM["X"]),
        "name": ACTIVE_STREAM["name"]
    }


@app.get("/api/timeline_overview")
def get_timeline_overview():
    """Returns overview arrays for rendering complete episode progression charts."""
    X = ACTIVE_STREAM["X"]
    Y = ACTIVE_STREAM["Y"]
    total = len(X)
    time_seconds = [int(i * config.TIME_STEP_SECONDS) for i in range(total)]

    # Sample top indicators for timeline overview
    syn_ratio = [float(X[i, config.FEATURE_NAMES.index("syn_flag_ratio")]) for i in range(total)]
    byte_rate = [float(X[i, config.FEATURE_NAMES.index("flow_bytes_per_sec")]) for i in range(total)]
    port_entropy = [float(X[i, config.FEATURE_NAMES.index("dst_port_entropy")]) for i in range(total)]
    stages = [int(y) for y in Y]

    return {
        "timestamps": time_seconds,
        "stages": stages,
        "syn_ratio": syn_ratio,
        "byte_rate": byte_rate,
        "port_entropy": port_entropy,
        "total_steps": total
    }


def generate_live_flows(stage_id, t):
    if stage_id == 0:
        return [
            {"src": "192.168.1.45", "dst": "142.250.190.46:443", "proto": "TCP / HTTPS", "bytes": "48.2 KB", "status": "Normal", "badge": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"},
            {"src": "192.168.1.12", "dst": "13.107.42.14:443", "proto": "TCP / Office365", "bytes": "12.8 KB", "status": "Normal", "badge": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"},
            {"src": "192.168.1.88", "dst": "8.8.8.8:53", "proto": "UDP / DNS", "bytes": "210 B", "status": "Normal", "badge": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"},
            {"src": "192.168.1.33", "dst": "104.16.132.229:443", "proto": "TCP / Cloudflare", "bytes": "320 KB", "status": "Normal", "badge": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"},
            {"src": "192.168.1.10", "dst": "192.168.1.1:80", "proto": "TCP / Internal HTTP", "bytes": "1.4 KB", "status": "Normal", "badge": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"},
        ]
    elif stage_id == 1:
        return [
            {"src": "45.33.32.156", "dst": "192.168.1.10:22", "proto": "TCP / Port Scan (SSH)", "bytes": "64 B", "status": "Suspicious Scan", "badge": "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"},
            {"src": "45.33.32.156", "dst": "192.168.1.10:445", "proto": "TCP / Port Scan (SMB)", "bytes": "64 B", "status": "Suspicious Scan", "badge": "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"},
            {"src": "45.33.32.156", "dst": "192.168.1.10:3389", "proto": "TCP / Port Scan (RDP)", "bytes": "64 B", "status": "Suspicious Scan", "badge": "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"},
            {"src": "45.33.32.156", "dst": "192.168.1.10:8080", "proto": "TCP / Port Scan (Web)", "bytes": "64 B", "status": "Suspicious Scan", "badge": "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"},
            {"src": "192.168.1.45", "dst": "142.250.190.46:443", "proto": "TCP / HTTPS", "bytes": "52.1 KB", "status": "Normal", "badge": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"},
        ]
    elif stage_id == 2:
        return [
            {"src": "45.33.32.156", "dst": "192.168.1.10:22", "proto": "SSH / Brute-Force Auth", "bytes": "1.2 KB", "status": "Attack in Progress", "badge": "bg-orange-500/10 text-orange-400 border border-orange-500/20"},
            {"src": "45.33.32.156", "dst": "192.168.1.10:22", "proto": "SSH / Password Guess", "bytes": "1.2 KB", "status": "Attack in Progress", "badge": "bg-orange-500/10 text-orange-400 border border-orange-500/20"},
            {"src": "45.33.32.156", "dst": "192.168.1.10:22", "proto": "SSH / Exploit Attempt", "bytes": "2.4 KB", "status": "Attack in Progress", "badge": "bg-orange-500/10 text-orange-400 border border-orange-500/20"},
            {"src": "192.168.1.10", "dst": "45.33.32.156:4444", "proto": "TCP / Remote Shell", "bytes": "890 B", "status": "Breach Detected", "badge": "bg-rose-500/20 text-rose-400 border border-rose-500/40"},
            {"src": "192.168.1.12", "dst": "13.107.42.14:443", "proto": "TCP / HTTPS", "bytes": "18.4 KB", "status": "Normal", "badge": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"},
        ]
    elif stage_id == 3:
        return [
            {"src": "192.168.1.10", "dst": "192.168.1.15:445", "proto": "SMB / Network Pivot", "bytes": "34.5 KB", "status": "Spreading Across Network", "badge": "bg-rose-500/10 text-rose-400 border border-rose-500/20"},
            {"src": "192.168.1.10", "dst": "192.168.1.18:135", "proto": "RPC / Remote Exec", "bytes": "12.1 KB", "status": "Spreading Across Network", "badge": "bg-rose-500/10 text-rose-400 border border-rose-500/20"},
            {"src": "192.168.1.10", "dst": "192.168.1.22:3389", "proto": "RDP / Credential Sweep", "bytes": "8.4 KB", "status": "Spreading Across Network", "badge": "bg-rose-500/10 text-rose-400 border border-rose-500/20"},
            {"src": "192.168.1.15", "dst": "45.33.32.156:4444", "proto": "TCP / Hacker Control Beacon", "bytes": "240 B", "status": "Beacon Active", "badge": "bg-purple-500/10 text-purple-400 border border-purple-500/20"},
            {"src": "192.168.1.88", "dst": "8.8.8.8:53", "proto": "UDP / DNS", "bytes": "220 B", "status": "Normal", "badge": "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"},
        ]
    else:
        return [
            {"src": "192.168.1.10", "dst": "185.220.101.5:443", "proto": "TCP / Data Exfiltration", "bytes": "48.2 MB", "status": "Stolen Data Upload!", "badge": "bg-purple-500/20 text-purple-300 font-bold border border-purple-500/50"},
            {"src": "192.168.1.15", "dst": "185.220.101.5:443", "proto": "TCP / Archive Stream", "bytes": "32.1 MB", "status": "Stolen Data Upload!", "badge": "bg-purple-500/20 text-purple-300 font-bold border border-purple-500/50"},
            {"src": "192.168.1.10", "dst": "185.220.101.5:443", "proto": "TCP / Encrypted Tunnel", "bytes": "89.4 MB", "status": "Stolen Data Upload!", "badge": "bg-purple-500/20 text-purple-300 font-bold border border-purple-500/50"},
            {"src": "45.33.32.156", "dst": "192.168.1.1:80", "proto": "TCP / DoS Packet Flood", "bytes": "12.8 MB", "status": "Denial of Service Spike", "badge": "bg-rose-500/20 text-rose-300 font-bold border border-rose-500/50"},
            {"src": "192.168.1.45", "dst": "142.250.190.46:443", "proto": "TCP / Connection Timeout", "bytes": "0 B", "status": "Service Degraded", "badge": "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"},
        ]


@app.get("/api/forecast_at_step")
def forecast_at_step(t: int = Query(..., ge=config.WINDOW_SIZE)):
    X = ACTIVE_STREAM["X"]
    Y = ACTIVE_STREAM["Y"]
    total = len(X)
    W = config.WINDOW_SIZE
    K = config.FORECAST_HORIZON

    if t >= total - K:
        t = total - K - 1

    x_window = torch.tensor(X[t - W:t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    curr_stage = int(Y[t - 1]) if len(Y) > t - 1 else 0
    actual_future_stage = int(Y[t + K - 1]) if len(Y) > t + K - 1 else 0

    with torch.no_grad():
        out = model(x_window, horizon=K)
        stage_logits = out["stage_logits"]
        pred_future_stage = int(torch.argmax(stage_logits[0, -1]).item())
        alarm_prob = float(out["early_warning_prob"][0, -1, 0].item())
        x_forecast = out["x_future_forecast"][0, -1].cpu().numpy().tolist()
        x_actual_future = X[t + K - 1].tolist()

        # Step by step predictions across horizon
        step_predictions = []
        for k in range(K):
            stg = int(torch.argmax(stage_logits[0, k]).item())
            prob = float(out["early_warning_prob"][0, k, 0].item())
            step_predictions.append({
                "step": k + 1,
                "seconds_ahead": (k + 1) * config.TIME_STEP_SECONDS,
                "predicted_stage_id": stg,
                "predicted_stage_name": config.STAGE_NAMES[stg],
                "threat_prob": prob
            })

    # Integrated Gradients XAI Explanation
    explanation = explainer.explain_future_stage(x_window, target_stage=pred_future_stage, forecast_step=-1)
    narrative = explainer.generate_narrative_explanation(explanation, stage_idx=pred_future_stage)

    # Convert explanation dict to sorted list of objects with friendly names
    attribution_list = [
        {
            "feature": k,
            "friendly_name": FRIENDLY_FEATURE_NAMES.get(k, k),
            "score": round(v, 2)
        }
        for k, v in list(explanation.items())[:6]
    ]

    # Proactive alert trigger condition
    lead_time_seconds = K * config.TIME_STEP_SECONDS
    is_early_warning = (pred_future_stage > curr_stage and alarm_prob >= 0.65)
    is_active_attack = (curr_stage > 0)

    return {
        "timestep": t,
        "current_time_seconds": t * config.TIME_STEP_SECONDS,
        "lead_time_seconds": lead_time_seconds,
        "current_stage_id": curr_stage,
        "current_stage_name": FRIENDLY_STAGE_NAMES[curr_stage],
        "predicted_future_stage_id": pred_future_stage,
        "predicted_future_stage_name": FRIENDLY_STAGE_NAMES[pred_future_stage],
        "actual_future_stage_id": actual_future_stage,
        "threat_probability": round(alarm_prob, 4),
        "is_early_warning": is_early_warning,
        "is_active_attack": is_active_attack,
        "horizon_steps": step_predictions,
        "feature_attributions": attribution_list,
        "soc_narrative": narrative,
        "simple_recommendation": SIMPLE_RECOMMENDATIONS.get(pred_future_stage, "Keep monitoring network traffic."),
        "live_flows": generate_live_flows(curr_stage, t),
        "telemetry_comparison": {
            "features": [FRIENDLY_FEATURE_NAMES.get(f, f) for f in config.FEATURE_NAMES],
            "actual": [round(v, 3) for v in x_actual_future],
            "forecasted": [round(v, 3) for v in x_forecast]
        }
    }


@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
        X, Y = adapter.transform_dataframe(df)
        if len(X) < config.WINDOW_SIZE + config.FORECAST_HORIZON + 2:
            raise HTTPException(status_code=400, detail="CSV must have at least 20 rows of flows.")
        ACTIVE_STREAM["X"] = X
        ACTIVE_STREAM["Y"] = Y
        ACTIVE_STREAM["name"] = f"Uploaded CSV ({file.filename})"
        return {
            "message": "CSV uploaded and parsed successfully",
            "filename": file.filename,
            "total_steps": len(X),
            "features_detected": [f for f in config.FEATURE_NAMES if f in df.columns]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_test_data")

DEMO_DATASETS = [
    {
        "id": "01_Full_Attack_Progression.csv",
        "title": "Sample 1: Complete 5-Stage Attack Progression",
        "description": "Full progression from Safe to Recon, Exploit, Spreading, and Data Theft.",
        "badge": "Full Attack Lifecycle"
    },
    {
        "id": "02_Normal_Safe_Traffic.csv",
        "title": "Sample 2: 100% Safe Normal Traffic",
        "description": "Healthy daily traffic to test false-alarm resistance (threat remains 0%).",
        "badge": "Safe / Baseline"
    },
    {
        "id": "03_Password_BruteForce_Attack.csv",
        "title": "Sample 3: Password Brute-Force Exploit",
        "description": "Port scan transitioning directly into rapid authentication brute force.",
        "badge": "Exploit Attack"
    },
    {
        "id": "04_DDoS_Flood_Attack.csv",
        "title": "Sample 4: DDoS Traffic Flood",
        "description": "Massive packet flood with SYN surge and bandwidth saturation.",
        "badge": "DDoS Flood"
    }
]


@app.get("/api/list_demo_files")
def list_demo_files():
    return DEMO_DATASETS


@app.get("/api/load_demo_file")
def load_demo_file(filename: str):
    file_path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Demo file not found.")
    X, Y = adapter.load_and_transform_csv(file_path)
    ACTIVE_STREAM["X"] = X
    ACTIVE_STREAM["Y"] = Y
    ACTIVE_STREAM["name"] = filename
    return {
        "message": f"Loaded demo file: {filename}",
        "filename": filename,
        "total_steps": len(X)
    }


@app.get("/api/download_demo_file")
def download_demo_file(filename: str):
    file_path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Demo file not found.")
    return FileResponse(file_path, filename=filename, media_type="text/csv")


# Static frontend files mount
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "SOC Dashboard API is running. Frontend index.html being built."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
