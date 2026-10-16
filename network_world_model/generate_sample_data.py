"""
Generator script for 4 distinct uploadable Network Traffic CSV datasets:
1. scenario_1_full_attack_killchain.csv  - Complete 5-stage cyber attack progression
2. scenario_2_benign_normal_traffic.csv   - 100% safe, clean network traffic
3. scenario_3_password_bruteforce.csv     - Port scan transitioning to auth brute-force
4. scenario_4_ddos_traffic_flood.csv      - High volume DoS/DDoS packet flood
"""
import numpy as np
import pandas as pd
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.killchain_simulator import NetworkKillChainSimulator

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_test_data")
os.makedirs(DEMO_DIR, exist_ok=True)

sim = NetworkKillChainSimulator(random_seed=123)

def create_csv(filename, X, Y, scenario_title):
    df = pd.DataFrame(X, columns=config.FEATURE_NAMES)
    df.insert(0, "Timestamp_Seconds", np.arange(len(df)) * config.TIME_STEP_SECONDS)
    df.insert(1, "True_Stage_Name", [config.STAGE_NAMES[y] for y in Y])
    df.insert(2, "True_Stage_ID", Y)
    
    file_path = os.path.join(DEMO_DIR, filename)
    df.to_csv(file_path, index=False)
    print(f"[GENERATED] [{scenario_title}]: {file_path} ({len(df)} rows)")
    return file_path

# 1. Full Attack Kill Chain (120 steps / 10 minutes)
X1, Y1 = sim.generate_episode(episode_type="full_attack", length=120)
create_csv("01_Full_Attack_Progression.csv", X1, Y1, "Scenario 1: Complete 5-Stage Cyber Attack")

# 2. Benign Normal Traffic (100 steps / 8.3 minutes)
X2, Y2 = sim.generate_episode(episode_type="benign_only", length=100)
create_csv("02_Normal_Safe_Traffic.csv", X2, Y2, "Scenario 2: 100% Safe Normal Background Traffic")

# 3. Password Brute Force & Initial Exploit
# Safe (30 steps) -> Recon Scan (30 steps) -> Exploit / Brute Force (40 steps)
length3 = 100
X3 = np.zeros((length3, config.NUM_FEATURES), dtype=np.float32)
Y3 = np.zeros(length3, dtype=np.int64)

for t in range(length3):
    if t < 30:
        X3[t] = sim._sample_stage_features(stage=0, step_in_stage=t, stage_duration=30)
        Y3[t] = 0
    elif t < 60:
        X3[t] = sim._sample_stage_features(stage=1, step_in_stage=t - 30, stage_duration=30)
        Y3[t] = 1
    else:
        X3[t] = sim._sample_stage_features(stage=2, step_in_stage=t - 60, stage_duration=40)
        Y3[t] = 2

create_csv("03_Password_BruteForce_Attack.csv", X3, Y3, "Scenario 3: Targeted Password Brute-Force Exploit")

# 4. Massive Denial of Service (DoS / DDoS) Flood
# Safe (25 steps) -> Rapid Probe (15 steps) -> Massive Flooding (60 steps)
length4 = 100
X4 = np.zeros((length4, config.NUM_FEATURES), dtype=np.float32)
Y4 = np.zeros(length4, dtype=np.int64)

for t in range(length4):
    if t < 25:
        X4[t] = sim._sample_stage_features(stage=0, step_in_stage=t, stage_duration=25)
        Y4[t] = 0
    elif t < 40:
        X4[t] = sim._sample_stage_features(stage=1, step_in_stage=t - 25, stage_duration=15)
        Y4[t] = 1
    else:
        # Extreme DoS spike: very high packets, high SYN, high bytes, short duration
        features = sim._sample_stage_features(stage=4, step_in_stage=t - 40, stage_duration=60)
        features[1] = np.clip(features[1] * 1.3, 0.0, 1.0) # Packet rate surge
        features[6] = np.clip(features[6] * 1.4, 0.0, 1.0) # SYN flood
        X4[t] = features
        Y4[t] = 4

create_csv("04_DDoS_Flood_Attack.csv", X4, Y4, "Scenario 4: Massive Denial of Service (DDoS) Flood")
