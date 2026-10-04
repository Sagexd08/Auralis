import json
from pathlib import Path
from .provenance import DatasetRecord


class Registry:
    """Dataset registry stored as one JSON record per line (diff-friendly)."""

    def __init__(self, path):
        self.path = Path(path)
        self._records: dict[str, DatasetRecord] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = DatasetRecord.from_json(json.loads(line))
                    self._records[r.dataset_id] = r

    def register(self, rec: DatasetRecord) -> None:
        self._records[rec.dataset_id] = rec

    def get(self, dataset_id: str) -> DatasetRecord:
        if dataset_id not in self._records:
            raise KeyError(f"dataset {dataset_id!r} is not in the registry (unregistered means unknown)")
        return self._records[dataset_id]

    def all(self):
        return [self._records[k] for k in sorted(self._records)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(r.to_json(), sort_keys=True) + "\n" for r in self.all()), encoding="utf-8")
