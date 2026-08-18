"""
Signal Processing Utilities for Insole Telemetry:
Windowing, Filtering, Pressure-Time Integral (PTI), Microvascular Fatigue Index (MFI),
and Sudomotor Impairment Index (SII).
"""

import numpy as np
from scipy.signal import butter, filtfilt

# Support both trapezoid (NumPy 2.0+) and trapz (older NumPy)
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
if _trapz is None:
    from scipy.integrate import trapezoid as _trapz


def butter_lowpass_filter(data: np.ndarray, cutoff: float = 5.0, fs: float = 20.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y


def compute_pressure_time_integral(fsr_data: np.ndarray, sampling_rate: float = 20.0) -> np.ndarray:
    """
    Computes Pressure-Time Integral (PTI) per sensor channel.
    PTI = sum(P(t) * dt)
    """
    dt = 1.0 / sampling_rate
    return _trapz(fsr_data, dx=dt, axis=0)


def compute_mfi(fsr_data: np.ndarray, temp_data: np.ndarray, rh_data: np.ndarray, sampling_rate: float = 20.0) -> np.ndarray:
    """
    Microvascular Fatigue Index (MFI):
    MFI(x, y) = ∫ [ P(x, y, τ) * (∂T(x, y, τ)/∂τ) * (1 - RH_norm(τ)) ] dτ
    """
    dt = 1.0 / sampling_rate
    rh_norm = rh_data / 100.0
    dT_dt = np.gradient(temp_data, dt, axis=0)

    if temp_data.ndim == 1 or temp_data.shape[1] == 1:
        dT_dt = np.repeat(dT_dt, 16, axis=1) if dT_dt.ndim > 1 else np.tile(dT_dt[:, None], (1, 16))

    if rh_norm.ndim == 1 or rh_norm.shape[1] == 1:
        rh_norm = np.repeat(rh_norm, 16, axis=1) if rh_norm.ndim > 1 else np.tile(rh_norm[:, None], (1, 16))

    integrand = fsr_data * dT_dt * (1.0 - rh_norm)
    mfi = _trapz(integrand, dx=dt, axis=0)
    return mfi


def compute_sii(rh_insole: np.ndarray, rh_ambient: float, gait_load: np.ndarray, t_insole: float, t_baseline: float) -> float:
    delta_rh = np.mean(rh_insole) - rh_ambient
    delta_load = np.ptp(gait_load) + 1e-6
    temp_ratio = t_insole / (t_baseline + 1e-6)

    sii = (delta_rh / delta_load) * np.exp(-temp_ratio)
    return float(sii)
