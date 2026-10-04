from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum

class LicenseState(str, Enum):
    UNKNOWN = "UNKNOWN"
    REJECTED = "REJECTED"
    NONCOMMERCIAL_ONLY = "NONCOMMERCIAL_ONLY"
    TRAINING_ONLY = "TRAINING_ONLY"
    REDISTRIBUTABLE = "REDISTRIBUTABLE"
    ELIGIBLE = "ELIGIBLE"

@dataclass
class DatasetRecord:
    dataset_id: str
    source: str
    revision: str
    license: str
    state: LicenseState = LicenseState.UNKNOWN
    training_allowed: bool = False
    commercial_allowed: bool = False
    redistribution_allowed: bool = False
    languages: list[str] = field(default_factory=list)
    hours: float = 0.0
    downloaded_at: str = ""
    checksum: str = ""
    notes: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @staticmethod
    def from_json(d: dict) -> "DatasetRecord":
        d = dict(d)
        d["state"] = LicenseState(d.get("state", "UNKNOWN"))
        return DatasetRecord(**d)

@dataclass
class SampleRecord:
    sample_id: str
    dataset_id: str
    path: str
    transcript: str
    language: str
    duration: float
    speaker_id: str = ""
    quality: float = 1.0
    sample_rate: int = 16000
