"""
PyTorch Dataset for Insole Time-Series Telemetry.
Processes windowed 16-channel FSR, thermistor temperature ΔT, and sudomotor RH%.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.utils.signal_processing import compute_mfi, compute_sii


class InsoleTelemetryDataset(Dataset):
    """
    Windowed time-series dataset for Stage 1 Pre-Ulcerative Early Warning Model.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 100,  # 5 seconds at 20Hz
        stride: int = 20,         # 1 second stride at 20Hz
        fs: float = 20.0
    ):
        """
        Args:
            data_dir: Directory containing session CSV files.
            window_size: Number of samples per window sequence.
            stride: Sliding window stride in samples.
            fs: Telemetry sampling rate (Hz).
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.fs = fs

        self.samples = []  # List of (window_tensor, label_scalar, mfi_vec, sii_val)
        self._load_and_window_data()

    def _load_and_window_data(self):
        if not os.path.exists(self.data_dir):
            return

        csv_files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith(".csv")
        ])

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)

            fsr_cols = [f"FSR_{i}" for i in range(16)]
            fsr_data = df[fsr_cols].values
            temp_data = df["Plantar_Temp_C"].values[:, None]
            rh_data = df["Sudomotor_RH_pct"].values[:, None]
            hazard_label = df["Hazard_Label"].max() if "Hazard_Label" in df else 0

            # Delta T relative to baseline
            baseline_temp = temp_data[0, 0]
            delta_t_data = temp_data - baseline_temp

            # Concatenate 18 channels: [FSR (16), DeltaT (1), RH (1)]
            features = np.hstack([fsr_data, delta_t_data, rh_data])

            n_samples = len(df)
            for start in range(0, n_samples - self.window_size + 1, self.stride):
                end = start + self.window_size
                window_feat = features[start:end]  # (window_size, 18)

                # Compute domain metrics for window
                window_fsr = fsr_data[start:end]
                window_temp = temp_data[start:end]
                window_rh = rh_data[start:end]

                mfi = compute_mfi(window_fsr, window_temp, window_rh, sampling_rate=self.fs)
                sii = compute_sii(window_rh, rh_ambient=45.0, gait_load=window_fsr.sum(axis=1),
                                  t_insole=float(window_temp[-1, 0]), t_baseline=baseline_temp)

                # Transpose to (C, T) = (18, window_size) for 1D-CNN
                feat_tensor = torch.from_numpy(window_feat.T).float()
                mfi_tensor = torch.from_numpy(mfi).float()
                sii_tensor = torch.tensor(sii, dtype=torch.float32)
                label_tensor = torch.tensor(hazard_label, dtype=torch.float32)

                self.samples.append((feat_tensor, label_tensor, mfi_tensor, sii_tensor))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]
