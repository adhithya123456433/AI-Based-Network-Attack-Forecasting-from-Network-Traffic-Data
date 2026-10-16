"""
Real-World Dataset Ingestion and Preprocessing Adapter.
Provides ingestion, column mapping, and normalization for real benchmark datasets:
- Direct CSV uploads (with standard schema)
- CSE-CIC-IDS2018
- CIC-IDS2017
- UNSW-NB15
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Dict, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Column mapping from standard CIC-IDS flow feature names to World Model feature schema
CIC_FEATURE_MAPPING = {
    "Flow Duration": "flow_duration_ms",
    "Fwd Packets/s": "fwd_pkts_per_sec",
    "Bwd Packets/s": "bwd_pkts_per_sec",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "SYN Flag Count": "syn_flag_ratio",
    "RST Flag Count": "rst_flag_ratio",
    "ACK Flag Count": "ack_flag_ratio",
    "Fwd IAT Mean": "fwd_iat_mean_ms",
    "Bwd IAT Mean": "bwd_iat_mean_ms"
}

# Mapping real attack string labels into Cyber Kill Chain stage IDs (0 to 4)
LABEL_STAGE_MAPPING = {
    "BENIGN": 0,
    "PortScan": 1,
    "Infiltration - Probe": 1,
    "FTP-Patator": 2,
    "SSH-Patator": 2,
    "Brute Force -Web": 2,
    "Brute Force -XSS": 2,
    "SQL Injection": 2,
    "Bot": 3,
    "Infiltration": 3,
    "DoS attacks-Hulk": 4,
    "DoS attacks-SlowHTTPTest": 4,
    "DDoS": 4,
    "DDoS attacks-LOIC-HTTP": 4
}


class RealNetworkDataAdapter:
    def __init__(self):
        self.feature_names = config.FEATURE_NAMES
        self.num_features = len(self.feature_names)

    def transform_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms any arbitrary DataFrame (uploaded CSV or loaded file) into:
        - X: (N, 16) array normalized in [0, 1]
        - Y: (N,) array of integer stage IDs (0 to 4)
        """
        # Clean column names
        df.columns = [c.strip() for c in df.columns]

        transformed = np.zeros((len(df), self.num_features), dtype=np.float32)

        # Case 1: Check if columns directly match our schema
        direct_matches = [f for f in self.feature_names if f in df.columns]
        if len(direct_matches) >= 8:
            for f in self.feature_names:
                if f in df.columns:
                    target_idx = self.feature_names.index(f)
                    vals = pd.to_numeric(df[f], errors='coerce').fillna(0).values
                    # If already normalized [0, 1] keep, else normalize
                    max_val = np.percentile(vals, 99) if len(vals) > 0 else 1.0
                    if max_val > 1.05:
                        vals = np.clip(vals / max_val, 0.0, 1.0)
                    transformed[:, target_idx] = vals
        else:
            # Case 2: Mapping from CIC-IDS schema
            for col, target_feat in CIC_FEATURE_MAPPING.items():
                if col in df.columns:
                    target_idx = self.feature_names.index(target_feat)
                    raw_vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                    p99 = np.percentile(raw_vals, 99) if np.percentile(raw_vals, 99) > 0 else 1.0
                    norm_vals = np.clip(raw_vals / p99, 0.0, 1.0)
                    transformed[:, target_idx] = norm_vals

        # Map labels if present
        labels = np.zeros(len(df), dtype=np.int64)
        if "True_Stage_ID" in df.columns:
            labels = pd.to_numeric(df["True_Stage_ID"], errors='coerce').fillna(0).values.astype(np.int64)
        elif "Label" in df.columns:
            for raw_lbl, stage_id in LABEL_STAGE_MAPPING.items():
                mask = df["Label"].astype(str).str.contains(raw_lbl, case=False, na=False)
                labels[mask] = stage_id

        return transformed, labels

    def load_and_transform_csv(self, csv_file_or_path: Union[str, any], max_rows: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads CSV from file path or file buffer.
        """
        df = pd.read_csv(csv_file_or_path, nrows=max_rows)
        return self.transform_dataframe(df)


if __name__ == "__main__":
    adapter = RealNetworkDataAdapter()
    test_csv = os.path.join(config.DATA_DIR, "sample_traffic_scenario.csv")
    if os.path.exists(test_csv):
        X, Y = adapter.load_and_transform_csv(test_csv)
        print("Successfully loaded test CSV. Shape:", X.shape, "Stages:", np.unique(Y))
