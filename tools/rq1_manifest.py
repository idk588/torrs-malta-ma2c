"""Record and check a manifest of the finished RQ1 run (Malta/signal_route).

Frozen files must never change; allowed files may still be updated by RQ1
follow-up scripts. Usage: python tools/rq1_manifest.py [--root DIR] record|check
Exit codes for check: 0 ok, 1 frozen mismatch, 2 manifest missing.
"""
import argparse
import csv
import hashlib
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

SR = "Malta/signal_route"
BACKUP_MANIFEST = SR + "/_rq1_final_backup/rq1_manifest.csv"
FROZEN_FILES = [
    "model/checkpoint-72000.data-00000-of-00001",
    "model/checkpoint-72000.index",
    "model/checkpoint-72000.meta",
    "train_reward.csv",
    "agent_reward.csv",
]
FROZEN_DIRS = ["eva_data", "greedy_data", "baseline_data/fixed_static",
               "baseline_data/fixed_rerouting", "baseline_data/actuated_static",
               "_rq1_final_backup"]
ALLOWED_DIRS = ["stats", "baseline_data/actuated_rerouting"]
ALLOWED_PREFIX = "ablation_data"
FIELDS = ["part", "path", "size", "mtime", "sha256"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_for(root, part, rel):
    p = root / rel
    st = p.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")
    return {"part": part, "path": rel, "size": st.st_size, "mtime": mtime,
            "sha256": sha256(p)}


def walk(root, rel_dir):
    d = root / rel_dir
    if not d.is_dir():
        return
    for dp, _, fns in os.walk(d):
        for fn in sorted(fns):
            yield (Path(dp) / fn).relative_to(root).as_posix()


def scan(root):
    sr = root / SR
    items = {}
    for f in FROZEN_FILES:
        if (sr / f).is_file():
            items[f"{SR}/{f}"] = "frozen"
    for d in FROZEN_DIRS:
        for rel in walk(root, f"{SR}/{d}"):
            items[rel] = "frozen"
    for d in ALLOWED_DIRS:
        for rel in walk(root, f"{SR}/{d}"):
            items[rel] = "allowed"
    if sr.is_dir():
        for e in sorted(sr.iterdir()):
            if e.is_dir() and e.name.startswith(ALLOWED_PREFIX):
                for rel in walk(root, f"{SR}/{e.name}"):
                    items[rel] = "allowed"
    items.pop(BACKUP_MANIFEST, None)
    return [row_for(root, items[r], r) for r in sorted(items)]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def record(root):
    qdir = root / "Malta" / "runs" / "_queue"
    qdir.mkdir(parents=True, exist_ok=True)
    mf = qdir / "rq1_manifest.csv"
    if mf.exists():
        with open(mf, encoding="utf-8") as f:
            first = f.readline()
        aside = qdir / "rq1_manifest_2026-11-08.csv"
        if not first.startswith("part,") and not aside.exists():
            mf.rename(aside)
    rows = scan(root)
    write_csv(mf, rows)
    bk = root / BACKUP_MANIFEST
    if bk.parent.is_dir():
        if bk.exists():
            os.chmod(bk, stat.S_IREAD | stat.S_IWRITE)
        write_csv(bk, rows)
        os.chmod(bk, stat.S_IREAD)
    nf = sum(r["part"] == "frozen" for r in rows)
    print(f"recorded {len(rows)} files ({nf} frozen, {len(rows) - nf} allowed) -> {mf}")
    return 0


def check(root):
    mf = root / "Malta" / "runs" / "_queue" / "rq1_manifest.csv"
    if not mf.exists():
        print(f"manifest missing: {mf}")
        return 2
    with open(mf, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "part" not in rows[0]:
        print("manifest is not in the new format (part,path,size,mtime,sha256)")
        return 2
    bad = nf = na = changed = 0
    for r in rows:
        p = root / r["path"]
        if p.is_file():
            try:
                cur = row_for(root, r["part"], r["path"])
                same = (str(cur["size"]) == r["size"] and cur["mtime"] == r["mtime"]
                        and cur["sha256"] == r["sha256"])
                why = "changed"
            except OSError as e:
                same, why = False, f"unreadable ({e})"
        else:
            same, why = False, "missing"
        if r["part"] == "frozen":
            nf += 1
            if not same:
                bad += 1
                print(f"MISMATCH frozen {why}: {r['path']}")
        else:
            na += 1
            if not same:
                changed += 1
                print(f"info: allowed file {why}: {r['path']}")
    print(f"frozen: {nf - bad}/{nf} unchanged, {bad} mismatched; "
          f"allowed: {na - changed}/{na} unchanged, {changed} changed")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("command", choices=["record", "check"])
    a = ap.parse_args(argv)
    root = Path(a.root)
    return record(root) if a.command == "record" else check(root)


if __name__ == "__main__":
    sys.exit(main())
