#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


def tee_run(cmd, log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== CMD ===\n{' '.join(cmd)}\n\n")
        f.write(f"=== START === {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        p.wait()
        f.write(f"\n=== END === {datetime.now().isoformat(timespec='seconds')}  rc={p.returncode}\n")
        return p.returncode

def read_bench_json(iter_dir: Path, tag: str) -> Optional[dict]:
    p = iter_dir / f"bench_{tag}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_summary_csv(rows: list[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "scene", "dataset",
        "model_path", "source_path",
        "npz_in", "npz_out",
        "finetune_rc", "decode_eval_rc",
        "PSNR", "SSIM", "LPIPS", "FPS", "PLY_MB", "num_gaussians",
        "skipped", "skip_reason",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in keys})

def maybe_warn_morton(npz_path: Path):
    # 仅做提示，不阻断
    try:
        import numpy as np
        z = np.load(npz_path, allow_pickle=False)
        morton_used = int(z["morton_used"]) if "morton_used" in z.files else None
        if morton_used == 1:
            print(f"[WARN] morton_used=1 in {npz_path.name}. "
                  f"如果你的 gs_codebook_finetune.py 没做同样的 morton perm 对齐，vqft 可能会崩。")
    except Exception:
        pass

def run_one(scene: str, dataset: str, dataset_root: Path, output_root: Path,
            iteration: int, tag_in: str, tag_out: str,
            steps: int, lr: float, batch: int, reassign_every: int, device: str,
            split: str, n_views: int, sh_degree: int,
            repo_root: Path, log_dir: Path) -> dict:

    model_path = output_root / scene
    iter_dir = model_path / "point_cloud" / f"iteration_{iteration}"

    ply = iter_dir / "point_cloud.ply"
    npz_in = iter_dir / f"point_cloud.{tag_in}.npz"
    npz_out = iter_dir / f"point_cloud.{tag_out}.npz"
    source_path = dataset_root / scene

    row = {
        "scene": scene,
        "dataset": dataset,
        "model_path": str(model_path),
        "source_path": str(source_path),
        "npz_in": str(npz_in),
        "npz_out": str(npz_out),
        "finetune_rc": None,
        "decode_eval_rc": None,
        "PSNR": None,
        "SSIM": None,
        "LPIPS": None,
        "FPS": None,
        "PLY_MB": None,
        "num_gaussians": None,
        "skipped": False,
        "skip_reason": None,
    }

    # checks
    if not iter_dir.exists():
        row["skipped"] = True
        row["skip_reason"] = f"missing iter_dir: {iter_dir}"
        return row
    if not ply.exists():
        row["skipped"] = True
        row["skip_reason"] = f"missing ply: {ply}"
        return row
    if not npz_in.exists():
        row["skipped"] = True
        row["skip_reason"] = f"missing npz_in: {npz_in}"
        return row
    if not source_path.exists():
        row["skipped"] = True
        row["skip_reason"] = f"missing source_path: {source_path}"
        return row

    print("\n" + "=" * 90)
    print(f"[{dataset}] scene={scene}")
    print(f"  model_path : {model_path}")
    print(f"  source_path: {source_path}")
    print(f"  npz_in     : {npz_in.name}")
    print(f"  npz_out    : {npz_out.name}")
    print("=" * 90)

    maybe_warn_morton(npz_in)

    ft_log = log_dir / f"ft_{scene}.log"
    eval_log = log_dir / f"eval_{scene}.log"

    # 1) finetune
    ft_cmd = [
        sys.executable, str(repo_root / "gs_codebook_finetune.py"),
        "--ply", str(ply),
        "--npz", str(npz_in),
        "--out", str(npz_out),
        "--steps", str(steps),
        "--lr", str(lr),
        "--batch", str(batch),
        "--reassign_every", str(reassign_every),
        "--device", device,
    ]
    print("[1/2] finetune ...")
    rc1 = tee_run(ft_cmd, ft_log, cwd=repo_root)
    row["finetune_rc"] = rc1
    if rc1 != 0:
        print(f"[ERROR] finetune failed: {scene}  rc={rc1}  log={ft_log}")
        return row

    # 2) decode + eval (gs_decode.py 自带 benchmark)
    # 注意：gs_decode.py 的默认 --bench 是 "benchmark_fps.py"，所以 cwd 必须是 repo_root
    eval_cmd = [
        sys.executable, str(repo_root / "gs_decode.py"),
        "--model_path", str(model_path),
        "--tag", tag_out,
        "-s", str(source_path),
        "--iteration", str(iteration),
        "--split", split,
        "--n_views", str(n_views),
        "--sh_degree", str(sh_degree),  # 作为 extra 传给 benchmark
    ]
    print("[2/2] decode + eval ...")
    rc2 = tee_run(eval_cmd, eval_log, cwd=repo_root)
    row["decode_eval_rc"] = rc2
    if rc2 != 0:
        print(f"[ERROR] decode/eval failed: {scene}  rc={rc2}  log={eval_log}")
        return row

    # parse bench json
    bench = read_bench_json(iter_dir, tag_out)
    if bench:
        for k in ["PSNR", "SSIM", "LPIPS", "FPS", "PLY_MB", "num_gaussians"]:
            if k in bench:
                row[k] = bench[k]

    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default="/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_sh2pvfine")
    ap.add_argument("--path_360", default="/root/autodl-tmp/gaussian-splatting/assets/360_v2")
    ap.add_argument("--path_tandt", default="/root/autodl-tmp/gaussian-splatting/assets/tandt_db/tandt")
    ap.add_argument("--path_db", default="/root/autodl-tmp/gaussian-splatting/assets/tandt_db/db")

    ap.add_argument("--iteration", type=int, default=5000)
    ap.add_argument("--tag_in", default="cb256")
    ap.add_argument("--tag_out", default="cb256_vqft")

    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--reassign_every", type=int, default=200)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--split", default="test")
    ap.add_argument("--n_views", type=int, default=50)
    ap.add_argument("--sh_degree", type=int, default=1)

    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--summary_csv", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_root = Path(args.output_root)

    log_dir = Path(args.log_dir) if args.log_dir else (output_root / "_vqft_logs")
    summary_csv = Path(args.summary_csv) if args.summary_csv else (log_dir / "summary.csv")

    tasks = [
        ("MipNeRF360", Path(args.path_360), ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]),
        ("TandT", Path(args.path_tandt), ["train", "truck"]),
        ("DeepBlending", Path(args.path_db), ["drjohnson", "playroom"]),
    ]

    rows = []
    for dataset, ds_root, scenes in tasks:
        print(f"\n### DATASET: {dataset}  root={ds_root}")
        for scene in scenes:
            r = run_one(
                scene=scene,
                dataset=dataset,
                dataset_root=ds_root,
                output_root=output_root,
                iteration=args.iteration,
                tag_in=args.tag_in,
                tag_out=args.tag_out,
                steps=args.steps,
                lr=args.lr,
                batch=args.batch,
                reassign_every=args.reassign_every,
                device=args.device,
                split=args.split,
                n_views=args.n_views,
                sh_degree=args.sh_degree,
                repo_root=repo_root,
                log_dir=log_dir,
            )
            if r.get("skipped"):
                print(f"[SKIP] {scene}: {r.get('skip_reason')}")
            rows.append(r)

    write_summary_csv(rows, summary_csv)
    print(f"\n== DONE ==\nLogs: {log_dir}\nSummary: {summary_csv}")

if __name__ == "__main__":
    main()
