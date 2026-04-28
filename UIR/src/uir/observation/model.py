from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


OBSERVATION_MODEL_KIND = "uir_observation_model"
OBSERVATION_MODEL_VERSION = 1


@dataclass(frozen=True)
class ObservationModel:
    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_value) or not np.isfinite(self.max_value):
            raise ValueError("observation model bounds must be finite")
        if self.max_value < self.min_value:
            raise ValueError("observation model max must be greater than or equal to min")

    def render(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, self.min_value, self.max_value)
        return np.rint(clipped).astype(np.float32)


def observation_model_path(volume_path: Path) -> Path:
    return volume_path.parent / f"{volume_path.name}.observation.json"


def write_observation_model(volume_path: Path, model: ObservationModel) -> Path:
    metadata_path = observation_model_path(volume_path)
    payload = {
        "kind": OBSERVATION_MODEL_KIND,
        "version": OBSERVATION_MODEL_VERSION,
        "min_value": float(model.min_value),
        "max_value": float(model.max_value),
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def read_observation_model(volume_path: Path) -> ObservationModel:
    metadata_path = observation_model_path(volume_path)
    if not metadata_path.exists():
        raise RuntimeError(
            f"Missing observation model metadata for {volume_path}. "
            "Regenerate this volume with the current pipeline."
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("kind") != OBSERVATION_MODEL_KIND:
        raise RuntimeError(f"Unexpected observation model kind in {metadata_path}")
    if payload.get("version") != OBSERVATION_MODEL_VERSION:
        raise RuntimeError(f"Unsupported observation model version in {metadata_path}")

    try:
        min_value = float(payload["min_value"])
        max_value = float(payload["max_value"])
    except KeyError as exc:
        raise RuntimeError(f"Observation model metadata is missing field {exc.args[0]} in {metadata_path}") from exc

    return ObservationModel(min_value=min_value, max_value=max_value)
