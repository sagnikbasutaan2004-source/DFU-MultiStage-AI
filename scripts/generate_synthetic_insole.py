"""
Synthetic Insole Telemetry Data Generator.
Generates 16-channel FSR piezoresistive pressure streams (20Hz),
plantar thermistors (1Hz interpolated), and sudomotor RH% (1Hz interpolated).
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_gait_cycle(num_steps: int = 100, fsr_channels: int = 16) -> np.ndarray:
    """
    Generates realistic 16-channel foot pressure stance phase profiles.
    Heel-strike → Mid-stance → Forefoot loading → Toe-off.
    """
    t = np.linspace(0, 1, num_steps)
    pressure_matrix = np.zeros((num_steps, fsr_channels))

    # Channels 0-3: Heel
    # Channels 4-7: Midfoot
    # Channels 8-13: Metatarsal heads (MTH 1-5)
    # Channels 14-15: Hallux / Toes
    heel_wave = np.sin(np.pi * t) * (t < 0.4)
    mid_wave = np.sin(np.pi * (t - 0.2)) * ((t >= 0.2) & (t <= 0.7))
    fore_wave = np.sin(np.pi * (t - 0.4)) * ((t >= 0.4) & (t <= 0.9))
    toe_wave = np.sin(np.pi * (t - 0.6)) * (t >= 0.6)

    for ch in range(4):
        pressure_matrix[:, ch] = heel_wave * (80.0 + np.random.uniform(-10, 10))
    for ch in range(4, 8):
        pressure_matrix[:, ch] = mid_wave * (40.0 + np.random.uniform(-5, 5))
    for ch in range(8, 14):
        pressure_matrix[:, ch] = fore_wave * (120.0 + np.random.uniform(-15, 15))
    for ch in range(14, 16):
        pressure_matrix[:, ch] = toe_wave * (90.0 + np.random.uniform(-10, 10))

    return np.maximum(0.0, pressure_matrix)


def generate_insole_session(
    duration_seconds: float = 60.0,
    fs: float = 20.0,
    has_hotspot: bool = False,
    hotspot_channel: int = 10
) -> pd.DataFrame:
    """
    Generates a full ambulatory session dataframe.
    """
    num_samples = int(duration_seconds * fs)
    t = np.arange(num_samples) / fs

    # Base FSR signals across multiple gait steps
    step_samples = int(1.0 * fs)  # 1 step per second
    num_steps = int(duration_seconds)

    fsr_data = np.zeros((num_samples, 16))
    single_step = generate_gait_cycle(step_samples, 16)

    for step in range(num_steps):
        start_idx = step * step_samples
        end_idx = min(start_idx + step_samples, num_samples)
        length = end_idx - start_idx
        fsr_data[start_idx:end_idx] = single_step[:length]

    # Add Gaussian noise
    fsr_data += np.random.normal(0, 2.0, fsr_data.shape)
    fsr_data = np.maximum(0.0, fsr_data)

    # Temperature profile (Base 30.5°C with optional thermal anomaly > 2.2°C rise)
    base_temp = 30.5 + np.linspace(0, 0.5, num_samples)
    if has_hotspot:
        # Simulate hyperthermic inflammatory hotspot
        hotspot_rise = 2.5 * (1.0 - np.exp(-t / 20.0))
        temp_data = base_temp + hotspot_rise
        fsr_data[:, hotspot_channel] *= 1.8  # Focal pressure overload at hotspot
    else:
        temp_data = base_temp

    # Sudomotor RH% (Normal: 40-60%, Impaired sudomotor: drops below 25%)
    if has_hotspot:
        rh_data = 55.0 - 25.0 * (t / duration_seconds) + np.random.normal(0, 1.0, num_samples)
    else:
        rh_data = 50.0 + 5.0 * np.sin(2 * np.pi * t / 15.0) + np.random.normal(0, 1.0, num_samples)

    columns = [f"FSR_{i}" for i in range(16)] + ["Plantar_Temp_C", "Sudomotor_RH_pct"]
    full_matrix = np.column_stack([fsr_data, temp_data, rh_data])

    df = pd.DataFrame(full_matrix, columns=columns)
    df.insert(0, "Time_s", t)
    df["Hazard_Label"] = 1 if has_hotspot else 0
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Insole Telemetry Dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw/insole_telemetry")
    parser.add_argument("--num_sessions", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating {args.num_sessions} synthetic insole telemetry sessions in {args.output_dir}...")

    for i in range(args.num_sessions):
        has_anomaly = (i % 2 == 1)  # 50% normal, 50% hyperthermia/overload
        df = generate_insole_session(duration_seconds=60.0, fs=20.0, has_hotspot=has_anomaly)
        file_path = os.path.join(args.output_dir, f"session_{i+1:03d}.csv")
        df.to_csv(file_path, index=False)

    print("Synthetic insole dataset generation complete! [OK]")
