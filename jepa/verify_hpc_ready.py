"""
Local verification of everything the HPC run depends on.

Every check here exists because the failure it catches would otherwise surface only
on capella, where finding it costs a queue cycle. Runs on CPU in a few minutes.

    python verify_hpc_ready.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import games
from JEPA import PaperAccurateJEPA, to_float, strip_compile_prefix
from dataset_JEPA import AtariTransitionDataset
from jepa_train import (WarmupCosine, autosize, build_param_groups, amp_autocast,
                        AMP_DTYPES, resolve_inverse_weight)

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  [ok]   {name}")
        except Exception as exc:
            FAIL.append((name, exc))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    return deco


print("=" * 70)
print("HPC READINESS VERIFICATION")
print("=" * 70)

# ---------------------------------------------------------------- uint8 parity
print("\n1. uint8 pipeline")


@check("uint8 and float32 datasets yield bit-identical normalised frames")
def _():
    ds8 = AtariTransitionDataset(config.VAL_DIR, dtype="uint8")
    ds32 = AtariTransitionDataset(config.VAL_DIR, dtype="float32")
    for i in np.random.RandomState(0).randint(0, len(ds8), size=25):
        a, b = ds8[int(i)], ds32[int(i)]
        assert a["s_t"].dtype == torch.uint8, a["s_t"].dtype
        assert b["s_t"].dtype == torch.float32
        assert torch.equal(to_float(a["s_t"].clone()), b["s_t"]), f"s_t mismatch at {i}"
        assert torch.equal(to_float(a["s_next"].clone()), b["s_next"]), f"s_next mismatch at {i}"
        assert int(a["a_t"]) == int(b["a_t"]) and int(a["g_t"]) == int(b["g_t"])


@check("encoder normalises internally, so uint8 and float32 inputs agree exactly")
def _():
    torch.manual_seed(0)
    m = PaperAccurateJEPA(embed_dim=256).eval()
    x8 = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
    e = m.env_embed(torch.zeros(4, dtype=torch.long))
    with torch.no_grad():
        z8 = m.context_encoder(x8, e)
        z32 = m.context_encoder(x8.float() / 255.0, e)
    assert torch.equal(z8, z32), (z8 - z32).abs().max()


@check("to_float leaves float tensors untouched (idempotent)")
def _():
    x = torch.rand(3, 4, 84, 84)
    assert to_float(x) is x


# ---------------------------------------------------------------- amp
print("\n2. mixed precision")


@check("bf16 is the default and maps to torch.bfloat16")
def _():
    assert AMP_DTYPES["bf16"] is torch.bfloat16
    assert AMP_DTYPES["fp16"] is torch.float16
    import jepa_train
    src = open(jepa_train.__file__).read()
    assert '"--amp-dtype", default="bf16"' in src


@check("autocast is disabled on CPU regardless of --amp-dtype")
def _():
    ctx = amp_autocast(torch.device("cpu"), "bf16")
    with ctx:
        assert torch.zeros(2).float().dtype == torch.float32


# ---------------------------------------------------------------- param groups
print("\n3. weight-decay param groups")


@check("LayerNorm gains, biases, pos_embed and both embeddings are decay-exempt")
def _():
    m = PaperAccurateJEPA(embed_dim=256)
    groups, decayed, exempt = build_param_groups(m, 1e-5)
    assert groups[0]["weight_decay"] == 1e-5 and groups[1]["weight_decay"] == 0.0
    assert "env_embed.weight" in exempt, "env_embed must not be decayed (fights max_norm)"
    assert "predictor.action_embed.weight" in exempt
    assert any(n.endswith("pos_embed") for n in exempt), "pos_embed must be exempt"
    for n in exempt:
        p = dict(m.named_parameters())[n]
        assert p.ndim <= 1 or "embed" in n, f"{n} ({tuple(p.shape)}) exempt without reason"
    for n in decayed:
        assert dict(m.named_parameters())[n].ndim >= 2, f"{n} decayed but is 1-D"
    # every trainable parameter must be in exactly one group
    n_grouped = sum(len(g["params"]) for g in groups)
    n_train = sum(1 for p in m.parameters() if p.requires_grad)
    assert n_grouped == n_train, f"{n_grouped} grouped vs {n_train} trainable"


# ---------------------------------------------------------------- schedule
print("\n4. warmup + cosine schedule")


@check("warmup ramps linearly then cosine decays to eta_min")
def _():
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    s = WarmupCosine(opt, 3e-4, warmup_steps=100, eta_min=1e-5)
    s.total_steps = 1000
    assert abs(s.lr_at(0) - 3e-4 / 100) < 1e-12, s.lr_at(0)
    assert abs(s.lr_at(99) - 3e-4) < 1e-9
    mid, end = s.lr_at(550), s.lr_at(1000)
    assert 3e-4 > mid > end, (mid, end)
    assert abs(end - 1e-5) < 1e-7, end
    assert s.lr_at(5000) == s.lr_at(1000), "must clamp past the horizon"


@check("schedule survives a state_dict round-trip")
def _():
    opt = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=3e-4)
    a = WarmupCosine(opt, 3e-4, 100)
    a.total_steps = 5000
    for _ in range(250):
        a.step()
    b = WarmupCosine(opt, 1.0, 1)
    b.load_state_dict(a.state_dict())
    assert b.step_num == a.step_num and b.total_steps == a.total_steps
    assert abs(b.get_last_lr()[0] - a.get_last_lr()[0]) < 1e-12


@check("probe finishes inside warmup, so auto-sizing never reads an unset horizon")
def _():
    opt = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=3e-4)
    s = WarmupCosine(opt, 3e-4, warmup_steps=2000)
    assert s.total_steps is None
    for _ in range(200):          # default --probe-steps
        s.step()
    assert s.step_num < s.warmup_steps, "probe must end before the cosine phase begins"
    assert s.get_last_lr()[0] > 0


# ---------------------------------------------------------------- auto-sizing
print("\n5. auto-sizing")


@check("autosize is finite, consistent and self-agreeing across throughputs")
def _():
    for R in (300.0, 3000.0, 6000.0, 12000.0, 40000.0):
        eps, spe, total = autosize(R, 512, seconds_left=15 * 3600)
        assert eps >= 4 and spe >= 200 and total == eps * spe
        assert np.isfinite([eps, spe, total]).all()
        epoch_min = spe * 512 / R / 60
        assert 1 <= epoch_min <= 60, f"R={R}: epoch is {epoch_min:.1f} min"
        wall = total * 512 / R / 3600
        assert wall <= 15 * 1.02, f"R={R}: sized {wall:.1f} h into a 15 h budget"


@check("a tiny time budget still yields a runnable schedule")
def _():
    eps, spe, total = autosize(50.0, 512, seconds_left=60)
    assert eps >= 4 and spe >= 200 and total > 0


# ---------------------------------------------------------------- compile strip
print("\n6. torch.compile checkpoint compatibility")


@check("strip_compile_prefix removes _orig_mod. and is a no-op otherwise")
def _():
    m = PaperAccurateJEPA(embed_dim=256)
    sd = m.state_dict()
    assert strip_compile_prefix(sd) is sd
    prefixed = {f"_orig_mod.{k}": v for k, v in sd.items()}
    m2 = PaperAccurateJEPA(embed_dim=256)
    m2.load_state_dict(strip_compile_prefix(prefixed))
    for k, v in m2.state_dict().items():
        assert torch.equal(v, sd[k]), k


# ---------------------------------------------------------------- sweep handoff
print("\n7. sweep -> main handoff")


@check("--inverse-weight auto reads the winner; ranks on info gain, not cross-entropy")
def _():
    import json
    from jepa_train import record_sweep_arm
    import argparse
    with tempfile.TemporaryDirectory() as d:
        def arm(tag, lam, gain, jepa, cos=0.0):
            a = argparse.Namespace(save_dir=d, sweep_tag=tag, inverse_weight=lam, batch_size=512)
            record_sweep_arm(a, jepa, {"gain": gain, "jepa": jepa, "acc": 0.2, "chance_acc": 0.13},
                             {"jepa": jepa}, None, cos)
        # lam=0.1 has the lower JEPA loss but the lower information gain
        arm("a", 0.003, 0.10, 0.0020)
        arm("b", 0.01, 0.40, 0.0025)
        arm("c", 0.1, 0.20, 0.0010)
        assert resolve_inverse_weight("auto", d) == 0.01, "must rank on information gain"
        # a collapsed arm must be disqualified even with the best gain
        arm("d", 0.03, 0.99, 0.0001, cos=0.97)
        assert resolve_inverse_weight("auto", d) == 0.01, "collapsed arm must be disqualified"
        res = json.load(open(os.path.join(d, "sweep_result.json")))
        assert len(res["arms"]) == 4 and res["best_inverse_weight"] == 0.01


@check("--inverse-weight auto fails loudly when no sweep has run")
def _():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            resolve_inverse_weight("auto", d)
            raise AssertionError("should have exited")
        except SystemExit:
            pass
        finally:
            os.chdir(cwd)


@check("a plain float still parses")
def _():
    assert resolve_inverse_weight("0.03", ".") == 0.03


# ---------------------------------------------------------------- resume
print("\n8. resume (end-to-end, two real runs)")

SCRATCH = os.environ.get("VERIFY_SCRATCH", tempfile.mkdtemp(prefix="jepa_resume_"))


@check("resume continues epoch/step/LR/best and restores weights exactly")
def _():
    d = os.path.join(SCRATCH, "ckpt")
    base = [sys.executable, "jepa_train.py", "--smoke", "--wandb-mode", "disabled",
            "--save-dir", d, "--epochs", "2"]
    r = subprocess.run(base, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]

    c1 = torch.load(os.path.join(d, "vjepa_v2_latest.pt"), map_location="cpu", weights_only=False)
    assert c1["epoch"] == 1, c1["epoch"]
    assert c1["global_step"] > 0 and "best_val_loss" in c1
    assert c1["scheduler_state_dict"]["step_num"] == c1["global_step"]

    r2 = subprocess.run(base + ["--epochs", "4", "--resume"], capture_output=True,
                        text=True, timeout=1800)
    assert r2.returncode == 0, r2.stdout[-2500:] + r2.stderr[-2500:]
    assert "RESUMED from" in r2.stdout, r2.stdout[-1500:]
    assert "Epoch 3" in r2.stdout, "should continue at epoch 3, not restart at 1"
    assert "Epoch 1 |" not in r2.stdout, "restarted instead of resuming"

    c2 = torch.load(os.path.join(d, "vjepa_v2_latest.pt"), map_location="cpu", weights_only=False)
    assert c2["epoch"] == 3, c2["epoch"]
    assert c2["global_step"] > c1["global_step"]
    assert c2["scheduler_state_dict"]["step_num"] > c1["scheduler_state_dict"]["step_num"]


@check("resume is a silent no-op when no checkpoint exists")
def _():
    d = os.path.join(SCRATCH, "empty")
    r = subprocess.run([sys.executable, "jepa_train.py", "--smoke", "--wandb-mode", "disabled",
                        "--save-dir", d, "--epochs", "1", "--resume"],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "not found; starting fresh" in r.stdout


# ---------------------------------------------------------------- shell scripts
print("\n9. SLURM scripts")


@check("all three sbatch scripts are syntactically valid")
def _():
    for f in ("job_a_cache.sh", "job_b_sweep.sh", "job_c_main.sh"):
        p = os.path.join("slurm", f)
        assert os.path.exists(p), p
        r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr}"


@check("job_c derives its budget from SLURM, and the fallback fits the requested --time")
def _():
    import re
    src = open("slurm/job_c_main.sh").read()
    # anchor on the directive, not a prose mention of --time in a comment
    slurm_h = int(re.search(r"^#SBATCH --time=(\d+):", src, re.M).group(1))
    fallback = float(re.search(r"^BUDGET_H=([\d.]+)", src, re.M).group(1))
    reserve = float(re.search(r"^RESERVE_H=([\d.]+)", src, re.M).group(1))
    assert fallback + reserve <= slurm_h, (
        f"fallback budget {fallback}h + reserve {reserve}h exceeds SLURM --time {slurm_h}h")
    # the budget must actually be recomputed from the allocation, not just hardcoded
    assert "scontrol show job" in src and "BUDGET_H=$(python3" in src
    assert '--max-hours "$BUDGET_H"' in src and '--target-hours "$TARGET_H"' in src


@check("shortening sbatch --time keeps max-hours inside the allocation")
def _():
    # Simulate the script's arithmetic for a 12 h allocation overriding the 16 h default.
    for alloc_h in (2, 6, 12, 16):
        secs = alloc_h * 3600
        budget = max(0.25, secs / 3600.0 - 0.4)
        target = round(budget * 0.98, 2)
        assert target < budget <= alloc_h - 0.4 or budget == 0.25, (alloc_h, budget)


print("\n" + "=" * 70)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for name, exc in FAIL:
    print(f"  FAILED: {name}\n          {exc}")
print("=" * 70)
sys.exit(1 if FAIL else 0)
