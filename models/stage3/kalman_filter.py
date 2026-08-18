"""
Stage 3: Extended Kalman Filter (EKF) for Longitudinal Wound Trajectory.
State vector x_k = [Area_k, Area_dot_k]^T
State transition: x_k = F * x_{k-1} + w_k
Measurement: z_k = H * x_k + v_k
"""

import numpy as np


class WoundKalmanFilter:
    """
    Kalman State Filter tracking wound area (mm²) and wound healing velocity dArea/dt (mm²/day).
    """

    def __init__(self, initial_area: float = 100.0, dt_days: float = 1.0, process_noise_std: float = 0.5, measurement_noise_std: float = 2.0):
        self.dt = dt_days

        # State vector x = [Area, Area_dot]^T
        self.x = np.array([[initial_area], [0.0]], dtype=np.float64)

        # State transition matrix F
        self.F = np.array([
            [1.0, self.dt],
            [0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H (we measure Area directly from segmentation)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Covariance matrix P
        self.P = np.eye(2, dtype=np.float64) * 10.0

        # Process noise Q
        q_var = process_noise_std ** 2
        self.Q = np.array([
            [0.25 * (self.dt ** 4), 0.5 * (self.dt ** 3)],
            [0.5 * (self.dt ** 3), self.dt ** 2]
        ], dtype=np.float64) * q_var

        # Measurement noise R
        self.R = np.array([[measurement_noise_std ** 2]], dtype=np.float64)

    def predict(self) -> np.ndarray:
        """
        Predict step: x_k = F * x_{k-1}, P_k = F * P_{k-1} * F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measured_area: float) -> np.ndarray:
        """
        Update step using measured wound surface area from SegFormer.
        """
        z = np.array([[measured_area]], dtype=np.float64)
        y = z - (self.H @ self.x)  # Innovation residual

        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return self.x

    def get_state(self) -> dict:
        return {
            'area_mm2': float(self.x[0, 0]),
            'healing_rate_mm2_per_day': float(self.x[1, 0])
        }
