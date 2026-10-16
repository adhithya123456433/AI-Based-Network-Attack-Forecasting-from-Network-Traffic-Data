"""
Configuration parameters for the AI-Based Network Attack Forecasting World Model.
"""
import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_storage")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Cyber Kill Chain Stages
STAGE_NAMES = [
    "Benign Normal",             # Stage 0: Regular benign traffic
    "Reconnaissance (Probe)",    # Stage 1: Port scan, vulnerability sweep
    "Initial Compromise",        # Stage 2: Exploit attempt, brute force
    "Lateral Movement",          # Stage 3: Internal pivoting, credential stuffing
    "Exfiltration & Impact"      # Stage 4: Data exfiltration / DoS attack
]
NUM_STAGES = len(STAGE_NAMES)

# Network Flow Feature Definitions (inspired by CIC-IDS / NetFlow)
FEATURE_NAMES = [
    "flow_duration_ms",          # Total duration of flow
    "fwd_pkts_per_sec",          # Forward packets per second
    "bwd_pkts_per_sec",          # Backward packets per second
    "flow_bytes_per_sec",        # Byte transmission rate
    "fwd_pkt_len_mean",          # Mean forward packet size
    "bwd_pkt_len_mean",          # Mean backward packet size
    "syn_flag_ratio",            # Ratio of SYN packets (high in scans/SYN floods)
    "rst_flag_ratio",            # Ratio of RST packets (connection rejections)
    "ack_flag_ratio",            # Ratio of ACK packets
    "fwd_iat_mean_ms",           # Forward inter-arrival time mean
    "bwd_iat_mean_ms",           # Backward inter-arrival time mean
    "dst_port_entropy",          # Port diversity (high during port sweeps)
    "payload_entropy",           # Payload randomness (high in encrypted exfil/tunneling)
    "failed_conn_ratio",         # Failed / half-open handshake ratio
    "byte_ratio_fwd_bwd",        # Skewness between outbound and inbound data
    "conn_density_window"        # Simultaneous connections from same source subnet
]
NUM_FEATURES = len(FEATURE_NAMES)

# Time Series & Forecasting Parameters
WINDOW_SIZE = 12                 # W: Observation window length (history timesteps)
FORECAST_HORIZON = 4             # K: Steps ahead to forecast into the future
TIME_STEP_SECONDS = 5            # Each timestep represents 5 seconds of aggregated flow

# World Model Hyperparameters
LATENT_DIM = 32                  # Dimension of compressed latent state z_t
HIDDEN_DIM = 64                  # Hidden size of dynamic recurrent / transition units
DYNAMICS_TYPE = "rssm_gru"       # Options: "rssm_gru", "transformer"
DROPOUT = 0.15

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 25
ALPHA_RECON = 1.0                # Weight for state reconstruction loss
BETA_TRANS = 0.5                 # Weight for latent transition consistency loss
GAMMA_STAGE = 2.0                # Weight for attack stage forecasting loss
