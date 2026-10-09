"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    transform = np.asarray(sensor.veh_to_sens)
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    p_s = transform[:3, :3] @ position + transform[:3, 3]

    if not np.isfinite(p_s).all():
        return False

    if getattr(sensor, "name", None) == "camera" and p_s[0] <= 1e-6:
        return False

    angle = float(np.arctan2(p_s[1], p_s[0]))
    min_fov = min(sensor.fov)
    max_fov = max(sensor.fov)
    return min_fov <= angle <= max_fov


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    transform = np.asarray(sensor.veh_to_sens)
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    p_s = transform[:3, :3] @ position + transform[:3, 3]

    if not np.isfinite(p_s).all() or p_s[0] <= 1e-6:
        raise ValueError(
            f"Invalid camera sensor coordinates {p_s.tolist()}: nonfinite or depth <= 1e-6"
        )

    depth = p_s[0]
    left = p_s[1]
    up = p_s[2]

    u = sensor.c_i - sensor.f_i * left / depth
    v = sensor.c_j - sensor.f_j * up / depth
    return np.asmatrix([[u], [v]])


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    z_mat = np.asmatrix([[float(z[0])], [float(z[1])]])
    R_mat = np.asmatrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2]))
    return {
        "z": z_mat,
        "R": R_mat,
        "sensor": sensor,
    }
