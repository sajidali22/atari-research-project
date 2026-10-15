"""Export the W&B record (the source of truth for all training runs) to artifacts/wandb_export/.

Usage:
    python evaluation/report_scripts/export_wandb.py inventory          # run tables for both projects
    python evaluation/report_scripts/export_wandb.py history RUN_ID ... # full history CSVs for given runs
    python evaluation/report_scripts/export_wandb.py history-all        # history for every non-trivial run

Credentials come from ~/.netrc or WANDB_API_KEY (never hard-code a key).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import wandb

ENTITY = "sajid-alisadruddin-"
PROJECTS = ["atari-universal-feature-extractor", "atari-vjepa-world-model"]
OUT = Path(__file__).resolve().parents[2] / "artifacts" / "wandb_export"

# Only scalar summaries are kept; histograms / images / gradients are dropped.
SKIP_PREFIXES = ("gradients/", "parameters/", "_wandb")


def _scalar_summary(run):
    out = {}
    for k, v in run.summary._json_dict.items():
        if k.startswith(SKIP_PREFIXES):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


def _flat_config(run):
    cfg = {}
    for k, v in run.config.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                cfg[f"{k}.{k2}"] = v2
        else:
            cfg[k] = v
    return cfg


def inventory():
    api = wandb.Api(timeout=60)
    OUT.mkdir(parents=True, exist_ok=True)
    for project in PROJECTS:
        rows, full = [], []
        for run in api.runs(f"{ENTITY}/{project}", order="+created_at"):
            summ, cfg = _scalar_summary(run), _flat_config(run)
            rows.append({
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "runtime_s": summ.get("_runtime"),
                "last_step": summ.get("_step"),
                "group": run.group,
                "job_type": run.job_type,
                "tags": ",".join(run.tags),
            })
            full.append({"id": run.id, "name": run.name, "state": run.state,
                         "created_at": run.created_at, "config": cfg, "summary": summ})
        pd.DataFrame(rows).to_csv(OUT / f"{project}_runs.csv", index=False)
        (OUT / f"{project}_runs_full.json").write_text(json.dumps(full, indent=1, default=str))
        print(f"{project}: {len(rows)} runs")


def history(project, run_ids):
    api = wandb.Api(timeout=120)
    hdir = OUT / "history" / project
    hdir.mkdir(parents=True, exist_ok=True)
    for rid in run_ids:
        target = hdir / f"{rid}.csv"
        if target.exists():
            print("skip", rid)
            continue
        run = api.run(f"{ENTITY}/{project}/{rid}")
        rows = [r for r in run.scan_history(page_size=10000)]
        df = pd.DataFrame(rows)
        keep = [c for c in df.columns if not c.startswith(SKIP_PREFIXES)]
        df = df[keep]
        # drop media/histogram columns (dict-valued)
        for c in list(df.columns):
            if df[c].map(lambda v: isinstance(v, dict)).any():
                df = df.drop(columns=c)
        df.to_csv(target, index=False)
        print(f"{rid}: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inventory":
        inventory()
    elif cmd == "history":
        project = sys.argv[2]
        history(project, sys.argv[3:])
    else:
        raise SystemExit(__doc__)
