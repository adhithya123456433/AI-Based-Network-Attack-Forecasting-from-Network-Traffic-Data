"""
Realistic Multi-Stage Cyber Kill Chain Network Flow Simulation Generator.
Synthesizes temporal network telemetry across:
- Stage 0: Benign Normal Traffic
- Stage 1: Reconnaissance (Port Scan / Network Probe)
- Stage 2: Initial Compromise (Exploit attempt / Brute Force)
- Stage 3: Lateral Movement (Credential pivoting / Subnet spread)
- Stage 4: Exfiltration & Impact (Massive data exfil or DoS spike)
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class NetworkKillChainSimulator:
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        self.feature_names = config.FEATURE_NAMES
        self.num_features = len(self.feature_names)

    def _sample_stage_features(self, stage: int, step_in_stage: int, stage_duration: int) -> np.ndarray:
        """
        Samples a 16-dimensional flow feature vector for a given attack stage,
        incorporating dynamic progression and smooth transitions.
        """
        progress = min(1.0, max(0.0, step_in_stage / float(stage_duration)))
        noise = self.rng.normal(0, 0.05, size=self.num_features)

        # Baseline benign values (normalized around 0.1 - 0.3)
        features = np.zeros(self.num_features, dtype=np.float32)

        # 0: flow_duration_ms
        # 1: fwd_pkts_per_sec
        # 2: bwd_pkts_per_sec
        # 3: flow_bytes_per_sec
        # 4: fwd_pkt_len_mean
        # 5: bwd_pkt_len_mean
        # 6: syn_flag_ratio
        # 7: rst_flag_ratio
        # 8: ack_flag_ratio
        # 9: fwd_iat_mean_ms
        # 10: bwd_iat_mean_ms
        # 11: dst_port_entropy
        # 12: payload_entropy
        # 13: failed_conn_ratio
        # 14: byte_ratio_fwd_bwd
        # 15: conn_density_window

        if stage == 0:
            # Benign Normal: Balanced, moderate traffic, low RST, low failed connections
            features[0] = 0.35 + self.rng.uniform(-0.08, 0.08)  # moderate flow duration
            features[1] = 0.20 + self.rng.uniform(-0.05, 0.05)  # fwd pkts
            features[2] = 0.22 + self.rng.uniform(-0.05, 0.05)  # bwd pkts
            features[3] = 0.25 + self.rng.uniform(-0.06, 0.06)  # flow bytes
            features[4] = 0.30 + self.rng.uniform(-0.05, 0.05)  # fwd pkt len
            features[5] = 0.40 + self.rng.uniform(-0.05, 0.05)  # bwd pkt len
            features[6] = 0.10 + self.rng.uniform(-0.03, 0.03)  # syn ratio
            features[7] = 0.03 + self.rng.uniform(-0.02, 0.02)  # rst ratio
            features[8] = 0.75 + self.rng.uniform(-0.05, 0.05)  # ack ratio
            features[9] = 0.30 + self.rng.uniform(-0.05, 0.05)  # fwd iat
            features[10] = 0.28 + self.rng.uniform(-0.05, 0.05) # bwd iat
            features[11] = 0.15 + self.rng.uniform(-0.04, 0.04) # low port entropy
            features[12] = 0.45 + self.rng.uniform(-0.05, 0.05) # normal payload entropy
            features[13] = 0.02 + self.rng.uniform(0.0, 0.02)   # low failed conn
            features[14] = 0.35 + self.rng.uniform(-0.05, 0.05) # balanced byte ratio
            features[15] = 0.15 + self.rng.uniform(-0.04, 0.04) # conn density

        elif stage == 1:
            # Reconnaissance (Port Scan / Probe):
            # High SYN ratio, high RST ratio (closed ports), high port entropy, short durations, small fwd pkts
            intensity = 0.5 + 0.5 * np.sin(progress * np.pi)
            features[0] = 0.08 + self.rng.uniform(0, 0.04)      # very short duration
            features[1] = 0.65 * intensity + 0.20               # fast scan packets
            features[2] = 0.08                                  # few backward packets
            features[3] = 0.15                                  # low bytes
            features[4] = 0.10                                  # tiny packets (SYN probes)
            features[5] = 0.05
            features[6] = 0.75 * intensity + 0.15               # high SYN
            features[7] = 0.55 * intensity + 0.10               # high RST
            features[8] = 0.20                                  # low ACK
            features[9] = 0.08                                  # very fast inter-arrival
            features[10] = 0.15
            features[11] = 0.85 * intensity + 0.10              # high destination port entropy
            features[12] = 0.20
            features[13] = 0.70 * intensity + 0.15              # failed handshakes
            features[14] = 0.80                                 # mostly fwd
            features[15] = 0.60 * intensity + 0.20

        elif stage == 2:
            # Initial Compromise (Exploit attempt / Brute Force):
            # Targeted port (low port entropy), high connection bursts, failed connections, payload entropy shift
            intensity = 0.6 + 0.4 * progress
            features[0] = 0.25 + self.rng.uniform(-0.04, 0.04)
            features[1] = 0.55 * intensity + 0.10               # repetitive auth / exploit packets
            features[2] = 0.30
            features[3] = 0.35 * intensity
            features[4] = 0.45 * intensity                      # specific exploit payload size
            features[5] = 0.25
            features[6] = 0.40 * intensity
            features[7] = 0.30 * intensity
            features[8] = 0.50
            features[9] = 0.18
            features[10] = 0.22
            features[11] = 0.12                                 # focused on specific target port (low entropy)
            features[12] = 0.65 * intensity + 0.15              # obfuscated shellcode / payload
            features[13] = 0.45 * intensity                     # rejected authentication attempts
            features[14] = 0.55
            features[15] = 0.70 * intensity                     # high connection density to victim

        elif stage == 3:
            # Lateral Movement (Internal Pivoting / C&C beacons):
            # Periodic bursts, moderate port entropy across internal IPs, high connection density
            periodic_pulse = 0.4 + 0.3 * np.cos(step_in_stage * 1.2)
            features[0] = 0.40 + self.rng.uniform(-0.05, 0.05)
            features[1] = 0.35 * periodic_pulse + 0.15
            features[2] = 0.30 * periodic_pulse + 0.15
            features[3] = 0.38 * periodic_pulse + 0.10
            features[4] = 0.40
            features[5] = 0.40
            features[6] = 0.25
            features[7] = 0.15
            features[8] = 0.60
            features[9] = 0.45 * periodic_pulse + 0.10          # beaconing periodicity
            features[10] = 0.40 * periodic_pulse + 0.10
            features[11] = 0.45                                 # moderate port diversity (SMB, RDP, SSH)
            features[12] = 0.55
            features[13] = 0.18
            features[14] = 0.48
            features[15] = 0.75 * progress + 0.20               # spreading across internal hosts

        elif stage == 4:
            # Exfiltration & Impact:
            # Massive outbound byte rate, high fwd/bwd ratio, high payload entropy (encrypted exfil)
            spike = 0.7 + 0.3 * np.sin(progress * np.pi / 2.0)
            features[0] = 0.85 * spike                          # long sustained connections
            features[1] = 0.80 * spike                          # high packet rate
            features[2] = 0.15                                  # low bwd packets (one-way stream)
            features[3] = 0.95 * spike                          # saturated bandwidth / high byte rate
            features[4] = 0.85 * spike                          # large MTU packets
            features[5] = 0.15
            features[6] = 0.12
            features[7] = 0.08
            features[8] = 0.70
            features[9] = 0.05                                  # continuous stream
            features[10] = 0.50
            features[11] = 0.08                                 # specific drop point / exfiltration server
            features[12] = 0.92 * spike                         # encrypted archive (high entropy)
            features[13] = 0.05
            features[14] = 0.95 * spike                         # extreme forward-to-backward byte skew
            features[15] = 0.65

        # Add Gaussian noise and clip to [0.0, 1.0]
        features = np.clip(features + noise, 0.0, 1.0)
        return features.astype(np.float32)

    def generate_episode(self, episode_type: str = "full_attack", length: int = 150) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a continuous time-series episode of network traffic flows.
        Returns:
            X: (length, num_features) matrix of flow telemetry
            Y: (length,) array of integer stage labels (0 to 4)
        """
        X = np.zeros((length, self.num_features), dtype=np.float32)
        Y = np.zeros(length, dtype=np.int64)

        if episode_type == "benign_only":
            for t in range(length):
                X[t] = self._sample_stage_features(stage=0, step_in_stage=t, stage_duration=length)
                Y[t] = 0
            return X, Y

        # Multi-stage kill chain:
        # Benign -> Recon -> Initial Compromise -> Lateral Movement -> Exfiltration
        # Randomize durations slightly
        benign_dur = self.rng.randint(20, 35)
        recon_dur = self.rng.randint(15, 25)
        exploit_dur = self.rng.randint(15, 25)
        lateral_dur = self.rng.randint(20, 30)
        
        stages_plan = [
            (0, benign_dur),
            (1, recon_dur),
            (2, exploit_dur),
            (3, lateral_dur),
            (4, length) # remaining goes to Stage 4
        ]

        current_t = 0
        for stage, duration in stages_plan:
            actual_dur = min(duration, length - current_t)
            if actual_dur <= 0:
                break
            for step in range(actual_dur):
                t = current_t + step
                X[t] = self._sample_stage_features(stage=stage, step_in_stage=step, stage_duration=actual_dur)
                Y[t] = stage
            current_t += actual_dur

        return X, Y

    def generate_dataset(self, num_episodes: int = 60, episode_length: int = 160) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a comprehensive dataset with a mix of:
        - Full attack kill chains
        - Benign network traffic
        - Partial / aborted attacks
        """
        all_X = []
        all_Y = []

        for i in range(num_episodes):
            roll = self.rng.rand()
            if roll < 0.25:
                ep_type = "benign_only"
            else:
                ep_type = "full_attack"
            X_ep, Y_ep = self.generate_episode(episode_type=ep_type, length=episode_length)
            all_X.append(X_ep)
            all_Y.append(Y_ep)

        return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.int64)


if __name__ == "__main__":
    sim = NetworkKillChainSimulator(random_seed=42)
    X, Y = sim.generate_dataset(num_episodes=5, episode_length=100)
    print(f"Generated simulated traffic shape: X={X.shape}, Y={Y.shape}")
    print(f"Features: {len(sim.feature_names)}")
    print(f"Unique stages present: {np.unique(Y)}")
