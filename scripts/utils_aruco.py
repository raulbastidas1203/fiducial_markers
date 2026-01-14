from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Calibration:
    dict_name: str
    id_left: int
    id_right: int
    d_min: float
    d_max: float
    method: str
    sources: list[str]
    timestamp: str

    @staticmethod
    def from_file(path: Path) -> "Calibration":
        data = json.loads(path.read_text())
        return Calibration(
            dict_name=data["dict"],
            id_left=int(data["id_left"]),
            id_right=int(data["id_right"]),
            d_min=float(data["d_min"]),
            d_max=float(data["d_max"]),
            method=data.get("method", "unknown"),
            sources=list(data.get("sources", [])),
            timestamp=data.get("timestamp", ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "dict": self.dict_name,
            "id_left": self.id_left,
            "id_right": self.id_right,
            "d_min": self.d_min,
            "d_max": self.d_max,
            "method": self.method,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))


@dataclass
class DetectionResult:
    d_px_raw: float
    conf: float
    centers: dict[int, np.ndarray]


class Smoother:
    def __init__(self, mode: str, alpha: float, window: int) -> None:
        self.mode = mode
        self.alpha = alpha
        self.window = window
        self._ema_value: float | None = None
        self._window_values: list[float] = []

    def update(self, value: float) -> float:
        if np.isnan(value):
            return np.nan
        if self.mode == "ema":
            if self._ema_value is None:
                self._ema_value = value
            else:
                self._ema_value = self.alpha * value + (1 - self.alpha) * self._ema_value
            return float(self._ema_value)
        self._window_values.append(value)
        if len(self._window_values) > self.window:
            self._window_values.pop(0)
        return float(np.mean(self._window_values))


def get_aruco_dictionary(dict_name: str) -> cv2.aruco_Dictionary:
    aruco = cv2.aruco
    return aruco.getPredefinedDictionary(getattr(aruco, dict_name))


def detect_markers(
    gray: np.ndarray,
    dictionary: cv2.aruco_Dictionary,
    params: cv2.aruco_DetectorParameters,
    id_left: int,
    id_right: int,
) -> DetectionResult:
    aruco = cv2.aruco
    corners_list, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

    centers: dict[int, np.ndarray] = {}
    if ids is not None and len(ids) > 0:
        for corners, mid in zip(corners_list, ids.flatten()):
            points = corners.reshape(4, 2)
            centers[int(mid)] = points.mean(axis=0)

    d_px_raw = np.nan
    conf = 0.0
    if id_left in centers and id_right in centers:
        p_left = centers[id_left]
        p_right = centers[id_right]
        d_px_raw = float(np.linalg.norm(p_left - p_right))
        conf = 1.0

    return DetectionResult(d_px_raw=d_px_raw, conf=conf, centers=centers)


def compute_width_norm(d_px_smooth: float, calib: Calibration) -> float:
    if np.isnan(d_px_smooth):
        return np.nan
    denom = calib.d_max - calib.d_min
    if abs(denom) <= 1e-6:
        return np.nan
    return float(np.clip((d_px_smooth - calib.d_min) / denom, 0.0, 1.0))


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    if values.size == 0:
        return values
    x = np.arange(values.size)
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        return values
    values[~mask] = np.interp(x[~mask], x[mask], values[mask])
    return values


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_calibration(
    dict_name: str,
    id_left: int,
    id_right: int,
    d_min: float,
    d_max: float,
    method: str,
    sources: list[str],
) -> Calibration:
    return Calibration(
        dict_name=dict_name,
        id_left=id_left,
        id_right=id_right,
        d_min=d_min,
        d_max=d_max,
        method=method,
        sources=sources,
        timestamp=now_iso(),
    )


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values, dtype=float), q))
