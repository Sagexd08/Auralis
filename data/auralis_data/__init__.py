"""Auralis Data Factory (M1): provenance records, licence gate, reproducible corpus builds."""
from .provenance import DatasetRecord, LicenseState, SampleRecord
from .gate import GateDecision, check_dataset, training_eligible
from .registry import Registry
from .corpus import CorpusConfig, LicenceGateError, build_corpus

__all__ = ["DatasetRecord", "SampleRecord", "LicenseState", "GateDecision", "check_dataset",
           "training_eligible", "Registry", "CorpusConfig", "LicenceGateError", "build_corpus"]
