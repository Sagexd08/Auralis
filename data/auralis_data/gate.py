from dataclasses import dataclass
from .provenance import DatasetRecord, LicenseState

_TRAIN = {LicenseState.TRAINING_ONLY, LicenseState.REDISTRIBUTABLE, LicenseState.ELIGIBLE}
_COMMERCIAL = {LicenseState.ELIGIBLE, LicenseState.REDISTRIBUTABLE}

@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str

def check_dataset(rec: DatasetRecord, commercial: bool = False) -> GateDecision:
    if rec.state in (LicenseState.UNKNOWN, LicenseState.REJECTED):
        return GateDecision(False, f"{rec.dataset_id}: licence state {rec.state.value}")
    if not rec.license.strip() or not rec.revision.strip():
        return GateDecision(False, f"{rec.dataset_id}: licence or revision not recorded")
    if not rec.training_allowed:
        return GateDecision(False, f"{rec.dataset_id}: training not allowed")
    if rec.state == LicenseState.NONCOMMERCIAL_ONLY:
        if commercial:
            return GateDecision(False, f"{rec.dataset_id}: noncommercial only")
        return GateDecision(True, "noncommercial training permitted")
    if commercial and (rec.state not in _COMMERCIAL or not rec.commercial_allowed):
        return GateDecision(False, f"{rec.dataset_id}: commercial use not cleared")
    if rec.state not in _TRAIN:
        return GateDecision(False, f"{rec.dataset_id}: state {rec.state.value}")
    return GateDecision(True, "ok")

def training_eligible(rec: DatasetRecord, commercial: bool = False) -> bool:
    return check_dataset(rec, commercial).allowed
