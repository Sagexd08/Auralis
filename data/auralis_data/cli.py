import argparse
import json
import sys
from .corpus import CorpusConfig, LicenceGateError, build_corpus
from .gate import check_dataset
from .provenance import DatasetRecord, LicenseState
from .registry import Registry

def main(argv=None):
    ap = argparse.ArgumentParser(prog="auralis-data")
    ap.add_argument("--registry", default="data/registry.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("dataset_id")
    r.add_argument("--source", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--license", default="")
    r.add_argument("--state", default="UNKNOWN", choices=[s.value for s in LicenseState])
    r.add_argument("--training", action="store_true")
    r.add_argument("--commercial", action="store_true")
    r.add_argument("--redistribution", action="store_true")
    r.add_argument("--languages", default="")
    c = sub.add_parser("check")
    c.add_argument("dataset_id")
    c.add_argument("--commercial", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("config")
    b.add_argument("--manifests", default="data/manifests")
    b.add_argument("--out", default="data/corpora")
    a = ap.parse_args(argv)

    reg = Registry(a.registry)
    if a.cmd == "register":
        reg.register(DatasetRecord(a.dataset_id, a.source, a.revision, a.license, LicenseState(a.state), a.training,
                                   a.commercial, a.redistribution, [x for x in a.languages.split(",") if x]))
        reg.save()
        print(f"registered {a.dataset_id} as {a.state}")
        return 0
    if a.cmd == "check":
        d = check_dataset(reg.get(a.dataset_id), a.commercial)
        print(("ALLOWED: " if d.allowed else "REFUSED: ") + d.reason)
        return 0 if d.allowed else 1
    with open(a.config, encoding="utf-8") as f:
        cfg = CorpusConfig(**json.load(f))
    try:
        meta = build_corpus(cfg, reg, a.manifests, a.out)
    except LicenceGateError as e:
        print(e, file=sys.stderr)
        return 2
    print(f"{meta['name']}: {meta['samples']} samples, {meta['hours']} h, sha256 {meta['sha256'][:16]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
