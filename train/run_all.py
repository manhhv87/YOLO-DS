"""Master experiment runner for the DS-YOLO paper.

Executes all steps from the PLAN.md in the correct order.
Each step is idempotent: if the expected output already exists it is skipped
(use --force to re-run).

Usage
-----
    cd train

    # Run everything from scratch
    python run_all.py

    # Run only specific steps
    python run_all.py --steps resplit train_all

    # Re-run even if outputs exist
    python run_all.py --force

    # Dry-run: print commands without executing
    python run_all.py --dry-run

Available steps (run in this order)
-------------------------------------
    resplit       Re-split in-house dataset by source image level
    train_all     Train the RGB baseline (Ultralytics pipeline)
    train_ds      Train DS-YOLO with train_ds.py
    fraction      Data-fraction study (H3): train rgb + ds-yolo at 10/25/50 %
    robustness    Robustness study: eval rgb vs ds-yolo under alignment perturbation
    soldef_val    Train RGB baseline on SolDef_A for external validation
    tables        Aggregate all results into LaTeX table rows
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Paths — all relative to train/
# ---------------------------------------------------------------------------
# In-house dataset (1 board, production cell)
INHOUSE_LEAKY    = HERE / "datasets" / "inhouse" / "leaky"   # original Roboflow export (DO NOT use for experiments)
GOLDEN           = HERE / "datasets" / "inhouse" / "golden" / "golden_inhouse.jpg"

# SolDef_A public dataset
SOLDEF_DATA      = HERE / "datasets" / "soldef" / "data.yaml"

# Generated datasets (created by scripts below)
DATASET_FIXED    = HERE / "dataset_fixed"               # resplit_inhouse.py output

RUNS_DETECT      = HERE / "runs" / "detect"
RUNS_DS          = HERE / "runs" / "ds_yolo"
RUNS_ROBUST      = HERE / "runs" / "robustness"
RUNS_FRACTION    = HERE / "runs" / "fraction"

# Training hyper-parameters (match paper Section VI-B and the actual checkpoints
# under runs/).  Override at runtime via --epochs / --imgsz / --batch / --size.
EPOCHS   = 100
IMGSZ    = 640
BATCH    = 4
SIZE     = "s"
# Multi-seed study: a +0.7pp mAP gap on an 11-image test set is only meaningful
# with variance. Override via --seeds 0 1 2. seed=0 keeps the canonical run names
# (rgb, ds_yolo, ...) for backward compatibility; seed>0 appends _s{seed}.
SEEDS    = [0]

# DS-YOLO-specific defaults:
#   FLIPLR=0.5: synced horizontal flip on capture, golden and bboxes (matches
#               the RGB baseline's fliplr=0.5 default — applied in train_ds.py
#               training loop, not via Ultralytics' YOLODataset).
#   NO_GRAD_GOLDEN=True: golden backbone pass runs under torch.no_grad() so its
#               activations are not retained for backprop.  Halves training-time
#               GPU memory at no quality cost.
#   ERASING=0.0: random erasing is DISABLED for the in-house comparison.
#               (The old 0.4 was a FAIRNESS BUG: Ultralytics 'erasing' applies
#               only to the classification pipeline, NOT detection, so the
#               RGB/Stack6 detection baselines get zero random-erasing while the
#               custom loop applied 40% — an uncontrolled extra augmentation on
#               the 'ours' variants only.)
FLIPLR         = "0.5"
NO_GRAD_GOLDEN = True
ERASING        = "0.0"   # disabled: detection baselines get none (see note above)
# EMA time-constant pinned to match Ultralytics ModelEMA (tau=2000); the custom
# loop's auto-shrink heuristic otherwise picks ~50 on short runs, giving the
# 'ours' variants a much faster EMA than the baselines.
EMA_TAU        = "2000"
# FREEZE_CRFM: DISABLED (0.0). Freezing CRFM for the first 70% of epochs
# GUARANTEES alpha stays 0 until the backbone has already converged on the
# capture alone — on this saturated single-board task that left alpha~0 for the
# whole run (CRFM never learned, DS-YOLO == RGB). To give CRFM any chance to
# learn, it must train from epoch 1 so its gradient exists while the task is
# still unsolved. (Re-enable only if you confirm CRFM still learns with it on.)
FREEZE_CRFM_FRAC = 0.0


# ---------------------------------------------------------------------------
# Shell helper
# ---------------------------------------------------------------------------

def run(cmd: list[str], dry: bool = False) -> None:
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n[run] {cmd_str}")
    if not dry:
        result = subprocess.run(cmd, cwd=str(HERE))
        if result.returncode != 0:
            print(f"[ERROR] command failed with code {result.returncode}")
            sys.exit(result.returncode)


def exists(path: Path) -> bool:
    return path.exists()


def skip(label: str, check: Path) -> bool:
    if check.exists():
        print(f"[skip] {label} — output already exists: {check}")
        return True
    return False


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

def step_resplit(dry: bool, force: bool) -> None:
    out = DATASET_FIXED / "data.yaml"
    if not force and exists(out):
        print(f"[skip] resplit — {out} exists")
        return
    run([sys.executable, "data/resplit_inhouse.py",
         "--src", str(INHOUSE_LEAKY),
         "--dst", str(DATASET_FIXED),
         "--val-frac", "0.15",
         "--test-frac", "0.15",
         "--seed", "0"], dry=dry)


def _train_variant(variant: str, data: Path, name: str | None = None,
                   fraction: float = 1.0, extra: list | None = None,
                   seed: int = 0, dry: bool = False) -> None:
    name = name or variant
    out  = RUNS_DETECT / name / "weights" / "best.pt"
    # Do not skip here — caller decides.
    cmd = [sys.executable, "train.py",
           "--variant",  variant,
           "--size",     SIZE,
           "--data",     str(data),
           "--epochs",   str(EPOCHS),
           "--imgsz",    str(IMGSZ),
           "--batch",    str(BATCH),
           "--seed",     str(seed),
           "--name",     name]
    if fraction < 1.0:
        cmd += ["--data-fraction", str(fraction)]
    if extra:
        cmd += extra
    run(cmd, dry=dry)


def step_train_all(dry: bool, force: bool) -> None:
    data_rgb = DATASET_FIXED / "data.yaml"

    # Only the RGB baseline is trained through the Ultralytics pipeline. The
    # other single-stream variants were removed because neither was a real
    # model: 'diff-only' had no diff preprocessing (plain RGB, byte-identical
    # test_results.json to rgb), and 'stack6' (6-ch) could not take effect
    # because yolo.train() rebuilds the model from its 3-ch YAML and discards
    # the first-conv patch, so it was also just a duplicate RGB run. The paper
    # compares RGB against DS-YOLO only.
    variants_data = [
        ("rgb",       data_rgb),
    ]
    for variant, data in variants_data:
        for seed in SEEDS:
            name = variant if seed == 0 else f"{variant}_s{seed}"
            out = RUNS_DETECT / name / "weights" / "best.pt"
            if not force and exists(out):
                print(f"[skip] train {name} — {out} exists")
                continue
            _train_variant(variant, data, name=name, seed=seed, dry=dry)


def _train_ds(name: str, fusion: str, dry: bool,
              save_dir: Path | None = None,
              data_fraction: float = 1.0,
              data: Path | None = None,
              seed: int = 0,
              extra: list | None = None) -> None:
    """Train one DS-YOLO variant (fusion=crfm or sub) via train_ds.py.

    Always passes ``--fliplr FLIPLR`` and ``--no-grad-golden`` so the run
    matches the paper's intended setup (synced flip, ~50 % less training
    GPU memory).  Override by appending to ``extra``.

    ``data`` lets the caller pass a fraction-subset data.yaml (shared across
    RGB and DS-YOLO) instead of the full dataset — see step_fraction.
    """
    cmd = [sys.executable, "train_ds.py",
           "--data",     str(data or (DATASET_FIXED / "data.yaml")),
           "--golden",   str(GOLDEN),
           "--variant",  SIZE,
           "--fusion",   fusion,
           "--epochs",   str(EPOCHS),
           "--imgsz",    str(IMGSZ),
           "--batch",    str(BATCH),
           "--fliplr",   FLIPLR,
           "--seed",     str(seed),
           "--name",     name,
           "--save-dir", str(save_dir or RUNS_DS)]
    if NO_GRAD_GOLDEN:
        cmd.append("--no-grad-golden")
    cmd += ["--erasing", ERASING]
    cmd += ["--ema-tau", EMA_TAU]
    if FREEZE_CRFM_FRAC > 0:
        cmd += ["--freeze-crfm", str(int(FREEZE_CRFM_FRAC * EPOCHS))]
    if data_fraction < 1.0:
        cmd += ["--data-fraction", str(data_fraction)]
    if extra:
        cmd += extra
    run(cmd, dry=dry)


def step_train_ds(dry: bool, force: bool) -> None:
    for seed in SEEDS:
        # DS-YOLO (CRFM fusion — main contribution)
        ds_name = "ds_yolo" if seed == 0 else f"ds_yolo_s{seed}"
        if force or not exists(RUNS_DS / ds_name / "weights" / "best.pt"):
            _train_ds(ds_name, fusion="crfm", dry=dry, seed=seed)
        else:
            print(f"[skip] train {ds_name} — output exists")


def step_fraction(dry: bool, force: bool) -> None:
    """Data-fraction study (H3): train rgb + ds-yolo at 10 / 25 / 50 %."""
    data_rgb = DATASET_FIXED / "data.yaml"

    # 100% is already covered by step_train_all (rgb) and step_train_ds (ds_yolo).
    # 10% added: the very-low-data regime is where the golden reference should
    # actually help (too few images for the capture-only model to memorise the
    # fixed layout) — this is the regime to watch the CRFM alpha and DS-vs-RGB gap.
    for frac in [0.10, 0.25, 0.50]:
        frac_tag = str(int(frac * 100))
        for seed in SEEDS:
            sfx = "" if seed == 0 else f"_s{seed}"

            # One shared, seed-fixed subset per (fraction, seed) so RGB and
            # DS-YOLO train on the IDENTICAL images. Both run at fraction=1.0.
            # (Previously RGB used Ultralytics first-N fraction and DS-YOLO a
            #  seeded randperm → different subsets, confounding the H3 claim.)
            subset_yaml = DATASET_FIXED / f"data_frac{frac_tag}_seed{seed}.yaml"
            if force or not exists(subset_yaml):
                run([sys.executable, "data/make_fraction_subset.py",
                     "--data",     str(data_rgb),
                     "--fraction", str(frac),
                     "--seed",     str(seed),
                     "--out-yaml", str(subset_yaml)], dry=dry)

            # RGB — fraction=1.0: the subset is baked into subset_yaml's list.
            name_rgb = f"rgb_{frac_tag}pct{sfx}"
            out_rgb  = RUNS_DETECT / name_rgb / "weights" / "best.pt"
            if force or not exists(out_rgb):
                _train_variant("rgb", subset_yaml, name=name_rgb, seed=seed, dry=dry)
            else:
                print(f"[skip] fraction {name_rgb} — {out_rgb} exists")

            # DS-YOLO — same subset_yaml, fraction left at 1.0.
            name_ds = f"ds_yolo_{frac_tag}pct{sfx}"
            out_ds  = RUNS_FRACTION / name_ds / "weights" / "best.pt"
            if force or not exists(out_ds):
                _train_ds(name_ds, fusion="crfm", dry=dry,
                          save_dir=RUNS_FRACTION, data=subset_yaml, seed=seed)
            else:
                print(f"[skip] fraction {name_ds} — {out_ds} exists")


def step_robustness(dry: bool, force: bool) -> None:
    """Robustness study: evaluate rgb vs ds-yolo under alignment perturbation (Table V)."""
    pairs = [
        ("rgb",     RUNS_DETECT / "rgb"    / "weights" / "best.pt",
                    DATASET_FIXED / "data.yaml",
                    RUNS_ROBUST / "rgb.csv",
                    False),
        ("ds-yolo", RUNS_DS / "ds_yolo" / "weights" / "best.pt",
                    DATASET_FIXED / "data.yaml",
                    RUNS_ROBUST / "ds_yolo.csv",
                    True),
    ]
    for variant, weights, data, out, needs_golden in pairs:
        if not force and exists(out):
            print(f"[skip] robustness {variant} — {out} exists")
            continue
        if not exists(weights):
            print(f"[skip] robustness {variant} — weights not found: {weights}")
            continue
        cmd = [sys.executable, "eval/eval_robustness.py",
               "--variant", variant,
               "--weights", str(weights),
               "--data",    str(data),
               "--imgsz",   str(IMGSZ),
               "--out",     str(out)]
        if needs_golden:
            cmd += ["--golden", str(GOLDEN)]
        run(cmd, dry=dry)


def step_soldef_val(dry: bool, force: bool) -> None:
    """Train RGB baseline on SolDef_A (external generalizability check)."""
    name = "rgb_soldef"
    out  = RUNS_DETECT / name / "weights" / "best.pt"
    if not force and exists(out):
        print(f"[skip] soldef_val — {out} exists")
        return
    if not SOLDEF_DATA.exists():
        print(f"[skip] soldef_val — SolDef data.yaml not found: {SOLDEF_DATA}")
        return
    # Patch data.yaml to use the correct absolute path for the current machine.
    # The original file may contain a hardcoded Windows path from the dev machine.
    import yaml as _yaml
    with SOLDEF_DATA.open() as f:
        _sd = _yaml.safe_load(f)
    _correct_path = str(SOLDEF_DATA.parent.resolve()).replace("\\", "/")
    if str(_sd.get("path", "")).replace("\\", "/") != _correct_path:
        _sd["path"] = _correct_path
        with SOLDEF_DATA.open("w") as f:
            _yaml.safe_dump(_sd, f, sort_keys=False)
        print(f"[soldef_val] patched data.yaml path -> {_correct_path}")
    # SolDef has multiple board layouts and NO golden reference, so the
    # alignment-preserving constraints used for the in-house dataset don't
    # apply.  --full-aug flips mosaic / translate / scale to YOLOv8 defaults.
    _train_variant("rgb", SOLDEF_DATA, name=name, dry=dry,
                   extra=["--full-aug"])


def step_soldef_finetune(dry: bool, force: bool) -> None:
    """Domain-specific pretraining ablation (Table II extension).

    Reuses the RGB-on-SolDef checkpoint produced by step_soldef_val as
    initialisation, then fine-tunes BOTH the RGB baseline and DS-YOLO on
    the in-house dataset for another EPOCHS epochs.

    Outputs (added to Table II via aggregate_results.py main):
      runs/detect/rgb_soldef_pre/             (variant tag: rgb-soldef-pre)
      runs/ds_yolo/ds_yolo_soldef_pre/        (variant tag: ds-yolo-soldef-pre)
    """
    pretrain_w = RUNS_DETECT / "rgb_soldef" / "weights" / "best.pt"
    if not pretrain_w.is_file():
        print(f"[skip] soldef_finetune -- pretrain weights not found: "
              f"{pretrain_w} (run step soldef_val first)")
        return

    data_inhouse = DATASET_FIXED / "data.yaml"

    # --- (a) RGB fine-tuned from SolDef pretrain --------------------------
    name_rgb = "rgb_soldef_pre"
    out_rgb  = RUNS_DETECT / name_rgb / "weights" / "best.pt"
    if force or not exists(out_rgb):
        _train_variant("rgb", data_inhouse, name=name_rgb, dry=dry,
                       extra=["--weights", str(pretrain_w)])
    else:
        print(f"[skip] soldef_finetune rgb -- {out_rgb} exists")

    # --- (b) DS-YOLO fine-tuned from SolDef pretrain ----------------------
    name_ds = "ds_yolo_soldef_pre"
    out_ds  = RUNS_DS / name_ds / "weights" / "best.pt"
    if force or not exists(out_ds):
        _train_ds(name_ds, fusion="crfm", dry=dry,
                  extra=["--pretrained", str(pretrain_w)])
    else:
        print(f"[skip] soldef_finetune ds-yolo -- {out_ds} exists")


def step_crfm_ablation(dry: bool, force: bool) -> None:
    """CRFM design study: justify the module's design choices.

    The full DS-YOLO (fuse p3,p4,p5 + sigmoid gate + learnable alpha) is already
    trained by step_train_ds as 'ds_yolo'. Here we train only the ablated
    variants and print a comparison table:
      * fusion location : P5 only vs P4+P5 (vs full P3+P4+P5 = ds_yolo)
      * gate            : additive injection without the learned gate
      * alpha           : fixed (=1) instead of learnable (tanh)
    """
    configs = [
        ("ds_crfm_p5",         ["--fuse-scales", "p5"]),
        ("ds_crfm_p4p5",       ["--fuse-scales", "p4,p5"]),
        ("ds_crfm_nogate",     ["--no-gate"]),
        ("ds_crfm_fixedalpha", ["--fixed-alpha"]),
    ]
    for name, extra in configs:
        out = RUNS_DS / name / "weights" / "best.pt"
        if not force and exists(out):
            print(f"[skip] crfm_ablation {name} — {out} exists")
            continue
        _train_ds(name, fusion="crfm", dry=dry, extra=extra)

    print("\n" + "=" * 60)
    print("CRFM DESIGN ABLATION")
    print("=" * 60)
    names = ["ds_yolo", "ds_crfm_p4p5", "ds_crfm_p5",
             "ds_crfm_nogate", "ds_crfm_fixedalpha"]
    run([sys.executable, "eval/aggregate_results.py", "ablation",
         "--ds-runs", str(RUNS_DS),
         "--names",   *names],
        dry=dry)


def step_harden(dry: bool, force: bool) -> None:
    """Inject controlled defects into dataset_fixed -> dataset_hard.

    Adds missing/shift/rotate NG instances (and keeps the originals) per split,
    breaking the mAP ceiling and balancing the OK/NG classes. NOTE: on a SINGLE
    board this does NOT make the golden necessary (a capture-only model can
    still memorise the fixed layout) — use step 'synth' for that.
    """
    hard = HERE / "dataset_hard"
    if force or not exists(hard / "data.yaml"):
        for split in ["train", "val", "test"]:
            run([sys.executable, "data/inject_defects.py",
                 "--images", str(DATASET_FIXED / split / "images"),
                 "--labels", str(DATASET_FIXED / split / "labels"),
                 "--out",    str(hard / split),
                 "--copies", "2", "--per-image", "3", "--include-original",
                 "--seed",   "0"], dry=dry)
        if not dry:
            (hard / "data.yaml").write_text(
                f"path: {hard.resolve().as_posix()}\n"
                f"train: train/images\nval: val/images\ntest: test/images\n"
                f"names:\n  0: OK\n  1: NG\nnc: 2\n", encoding="utf-8")
            print(f"[harden] wrote {hard / 'data.yaml'}")
    # Train RGB + DS-YOLO on the harder set (single fixed golden), over all
    # SEEDS so the row can be reported as mean+/-std like the data-fraction rows.
    for seed in SEEDS:
        rgb_name = "rgb_hard" if seed == 0 else f"rgb_hard_s{seed}"
        if force or not exists(RUNS_DETECT / rgb_name / "weights" / "best.pt"):
            _train_variant("rgb", hard / "data.yaml", name=rgb_name, seed=seed, dry=dry)
        else:
            print(f"[skip] train {rgb_name} — output exists")

        ds_name = "ds_yolo_hard" if seed == 0 else f"ds_yolo_hard_s{seed}"
        if force or not exists(RUNS_DS / ds_name / "weights" / "best.pt"):
            _train_ds(ds_name, fusion="crfm", dry=dry, data=hard / "data.yaml", seed=seed)
        else:
            print(f"[skip] train {ds_name} — output exists")


def step_synth(dry: bool, force: bool) -> None:
    """Synthetic REFERENCE-GUIDED benchmark: the layout varies per sample so the
    golden is NECESSARY (a capture-only model cannot tell a missing-NG slot from
    a legitimately-absent one). This is the decisive test of whether CRFM can
    learn: watch the [CRFM alpha] log and whether DS-YOLO (--golden-dir) beats
    the RGB baseline (which never sees the reference).

    Runs over all SEEDS: each seed regenerates an INDEPENDENT board-disjoint
    benchmark (different generation seed) AND trains with that seed, so Table II
    can be reported as mean+/-std over both the synthetic generation and the
    training randomness.
    """
    for seed in SEEDS:
        synth = HERE / ("dataset_synth" if seed == 0 else f"dataset_synth_s{seed}")
        if force or not exists(synth / "data.yaml"):
            run([sys.executable, "data/make_synthetic_refguided.py",
                 "--images", str(DATASET_FIXED / "train" / "images"),
                 "--labels", str(DATASET_FIXED / "train" / "labels"),
                 "--out", str(synth), "--n", "1500",
                 "--keep", "0.7", "--defect-rate", "0.25",
                 "--seed", str(seed)], dry=dry)
        # RGB baseline (capture only — expected to struggle on missing-NG).
        rgb_name = "rgb_synth" if seed == 0 else f"rgb_synth_s{seed}"
        if force or not exists(RUNS_DETECT / rgb_name / "weights" / "best.pt"):
            _train_variant("rgb", synth / "data.yaml", name=rgb_name, seed=seed, dry=dry)
        else:
            print(f"[skip] train {rgb_name} — output exists")
        # DS-YOLO with PER-SAMPLE golden (the reference it needs).
        ds_name = "ds_yolo_synth" if seed == 0 else f"ds_yolo_synth_s{seed}"
        if force or not exists(RUNS_DS / ds_name / "weights" / "best.pt"):
            _train_ds(ds_name, fusion="crfm", dry=dry, seed=seed,
                      data=synth / "data.yaml",
                      extra=["--golden-dir", str(synth)])
        else:
            print(f"[skip] train {ds_name} — output exists")
    print("\n[synth] Compare runs/detect/rgb_synth* vs runs/ds_yolo/ds_yolo_synth*.")
    print("[synth] If DS-YOLO >> RGB and [CRFM alpha] grew, CRFM works when the "
          "golden is required.")


def step_tables(dry: bool, force: bool = False) -> None:  # `force` accepted for uniform calling convention; tables always re-run
    """Print all LaTeX table rows from aggregated results."""
    print("\n" + "=" * 60)
    print("TABLE II — Main results")
    print("=" * 60)
    # Order must match the results table in the paper. SolDef-pretrain rows are
    # appended only if their checkpoints exist (step_soldef_finetune has run).
    # The paper compares RGB vs DS-YOLO; the stack6/f-sub/diff-only variants were
    # removed (see step_train_all / step_train_ds) so they are not tabulated.
    table2_variants = ["rgb", "ds-yolo"]
    if (RUNS_DETECT / "rgb_soldef_pre" / "weights" / "best.pt").is_file():
        table2_variants.append("rgb-soldef-pre")
    if (RUNS_DS / "ds_yolo_soldef_pre" / "weights" / "best.pt").is_file():
        table2_variants.append("ds-yolo-soldef-pre")
    run([sys.executable, "eval/aggregate_results.py", "main",
         "--runs",     str(RUNS_DETECT),
         "--ds-runs",  str(RUNS_DS),
         "--variants", *table2_variants],
        dry=dry)

    if len(SEEDS) > 1:
        print("\n--- Table II (mean +/- std over seeds) ---")
        run([sys.executable, "eval/aggregate_results.py", "mainstats",
             "--runs",     str(RUNS_DETECT),
             "--ds-runs",  str(RUNS_DS),
             "--variants", *table2_variants,
             "--seeds",    *[str(s) for s in SEEDS]],
            dry=dry)

    print("\n" + "=" * 60)
    print("TABLE III — Per-class mAP")
    print("=" * 60)
    run([sys.executable, "eval/aggregate_results.py", "perclass",
         "--runs",     str(RUNS_DETECT),
         "--ds-runs",  str(RUNS_DS),
         "--variants", "rgb", "ds-yolo",
         "--data",     str(DATASET_FIXED / "data.yaml"),
         "--golden",   str(GOLDEN),
         "--imgsz",    str(IMGSZ),
         "--split",    "test"],
        dry=dry)

    print("\n" + "=" * 60)
    print("TABLE IV — Latency")
    print("=" * 60)
    rgb_weights = RUNS_DETECT / "rgb" / "weights" / "best.pt"
    if exists(rgb_weights):
        run([sys.executable, "eval/aggregate_results.py", "latency",
             "--weights",    str(rgb_weights),
             "--imgsz",      str(IMGSZ),
             "--channels",   "3",
             "--runs-count", "200"],
            dry=dry)
    else:
        print("[skip] latency rgb — weights not found")

    print("\n" + "=" * 60)
    print("TABLE V — Robustness")
    print("=" * 60)
    # Table V compares RGB baseline vs DS-YOLO under alignment perturbation.
    robust_csvs = [
        (RUNS_ROBUST / "rgb.csv",      "RGB baseline"),
        (RUNS_ROBUST / "ds_yolo.csv",  r"DS-YOLO (ours)"),
    ]
    avail = [(str(p), lbl) for p, lbl in robust_csvs if p.exists()]
    if len(avail) >= 2:
        run([sys.executable, "eval/aggregate_results.py", "robust",
             "--csvs",   *[p for p, _ in avail],
             "--labels", *[lbl for _, lbl in avail]],
            dry=dry)
    else:
        print("[skip] robustness CSVs not found yet")


def step_figures(dry: bool, force: bool = False) -> None:  # `force` accepted for uniform calling convention; figures always re-run
    """Generate paper figures (requires trained weights)."""
    data_yaml = DATASET_FIXED / "data.yaml"
    rgb_w     = RUNS_DETECT / "rgb"    / "weights" / "best.pt"
    ds_w      = RUNS_DS     / "ds_yolo"/ "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("FIGURE 2 — Data fraction (H3)")
    print("=" * 60)
    run([sys.executable, "eval/plot_data_fraction.py"], dry=dry)

    print("\n" + "=" * 60)
    print("FIGURE 3 — Golden / Aligned / Diff")
    print("=" * 60)
    cmd_fig3 = [sys.executable, "eval/plot_qual_figures.py",
                "--data",     str(data_yaml),
                "--golden",   str(GOLDEN),
                "--imgsz",    str(IMGSZ),
                "--fig3-only"]
    run(cmd_fig3, dry=dry)

    print("\n" + "=" * 60)
    print("FIGURE 4 — RGB vs DS-YOLO detection")
    print("=" * 60)
    if exists(rgb_w) and exists(ds_w):
        cmd_fig4 = [sys.executable, "eval/plot_qual_figures.py",
                    "--data",     str(data_yaml),
                    "--golden",   str(GOLDEN),
                    "--imgsz",    str(IMGSZ),
                    "--rgb",      str(rgb_w),
                    "--ds-yolo",  str(ds_w)]
        run(cmd_fig4, dry=dry)
    else:
        print(f"[skip] Figure 4 — weights not found (need rgb + ds_yolo from train_ds)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_STEPS = [
    "resplit", "train_all", "train_ds",
    "fraction", "robustness", "soldef_val", "soldef_finetune",
    "tables", "figures",
]
# Optional steps: selectable via --steps but NOT run by the default (no-arg) run.
OPTIONAL_STEPS = ["crfm_ablation", "harden", "synth"]

STEP_FN = {
    "resplit":          step_resplit,
    "train_all":        step_train_all,
    "train_ds":         step_train_ds,
    "fraction":         step_fraction,
    "robustness":       step_robustness,
    "soldef_val":       step_soldef_val,
    "soldef_finetune":  step_soldef_finetune,
    "crfm_ablation":    step_crfm_ablation,
    "harden":           step_harden,
    "synth":            step_synth,
    "tables":           step_tables,
    "figures":          step_figures,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps",   nargs="+", default=ALL_STEPS,
                   choices=ALL_STEPS + OPTIONAL_STEPS,
                   help="Which steps to run (default: all non-optional). "
                        "Optional: crfm_ablation.")
    p.add_argument("--force",   action="store_true",
                   help="Re-run even if outputs already exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")
    # Hyper-parameters (override the constants at top of file)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz",  type=int, default=None)
    p.add_argument("--batch",  type=int, default=None)
    p.add_argument("--size",   default=None, choices=["n", "s", "m", "l", "x"])
    p.add_argument("--seeds",  type=int, nargs="+", default=None,
                   help="Seeds for the multi-seed study, e.g. --seeds 0 1 2 "
                        "(default: [0]). Runs are suffixed _s{seed} for seed>0.")
    args = p.parse_args()

    # Apply overrides to module-level constants so all step functions pick them up.
    global EPOCHS, IMGSZ, BATCH, SIZE, SEEDS
    if args.epochs is not None: EPOCHS = args.epochs
    if args.imgsz  is not None: IMGSZ  = args.imgsz
    if args.seeds  is not None: SEEDS  = args.seeds
    if args.batch  is not None: BATCH  = args.batch
    if args.size   is not None: SIZE   = args.size

    print(f"[run_all] steps: {args.steps}")
    print(f"[run_all] epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}  size={SIZE}")
    print(f"[run_all] force={args.force}  dry={args.dry_run}")

    # Auto-prepend 'resplit' if the chosen steps need dataset_fixed/ but it does
    # not exist yet (e.g. a fresh Colab session that lost the generated split).
    # resplit regenerates it from datasets/inhouse/leaky/ (committed to the repo).
    _needs_fixed = {"train_all", "train_ds", "fraction", "robustness",
                    "soldef_finetune", "crfm_ablation", "harden", "synth",
                    "tables", "figures"}
    steps = list(args.steps)
    if (set(steps) & _needs_fixed) and "resplit" not in steps \
            and not exists(DATASET_FIXED / "data.yaml"):
        print("[run_all] dataset_fixed/ not found — prepending 'resplit' "
              "(regenerates the source-level split from datasets/inhouse/leaky/).")
        steps = ["resplit"] + steps

    for step in steps:
        print(f"\n{'='*60}")
        print(f"STEP: {step}")
        print(f"{'='*60}")
        STEP_FN[step](dry=args.dry_run, force=args.force)

    print("\n[run_all] DONE")


if __name__ == "__main__":
    main()
