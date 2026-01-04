#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


def tee_run(cmd, log_path: Path, cwd: Optional[Path] = None) -> int:
    """
    Run a command, stream stdout/stderr to console and also write into log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== CMD ===\n{' '.join(cmd)}\n\n")
        f.write(f"=== START === {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert p.stdout is not None
        for line in p.stdout:
            # print to console
            print(line, end="")
            # write to file
            f.write(line)
        p.wait()

        f.write(f"\n=== END === {datetime.now().isoformat(timespec='seconds')}  rc={p.returncode}\n")
        f.flush()
        return p.returncode

def find_bench_json(iter_dir: Path, tag: str) -> Optional[Path]:
    """
    gs_decode.py 通常会写 bench_{tag}.json 到 iteration 目录（但也可能不同名字）。
    这里优先找 bench_{tag}.json，其次在目录里模糊找包含 tag 的 bench*.json。
    """
    p1 = iter_dir / f"bench_{tag}.json"
    if p1.exists():
        return p1
    # fallback: find any bench*.json containing tag
    cand = sorted(iter_dir.glob("bench_*.json"))
    for p in cand:
        if tag in p.name:
            return p
    return None

def run_scene(scene: str, dataset_root: Path, output_root: Path,
              it: int, tag_in: str, tag_out: str,
              steps: int, lr: float, batch: int, reassign_every: int, device: str,
              split: str, n_views: int, sh_degree: int,
              bench_script: Path,
              repo_root: Path,
              log_dir: Path,
              dry_run: bool = False) -> dict:
    """
    Return a dict with basic result info for summary.
    """
    model_path = output_root / scene
    iter_dir = model_path / "point_cloud" / f"iteration_{it}"

    ply = iter_dir / "point_cloud.ply"
    npz_in = iter_dir / f"point_cloud.{tag_in}.npz"
    npz_out = iter_dir / f"point_cloud.{tag_out}.npz"
    source_path = dataset_root / scene

    info = {
        "scene": scene,
        "model_path": str(model_path),
        "source_path": str(source_path),
        "npz_in": str(npz_in),
        "npz_out": str(npz_out),
        "finetune_rc": None,
        "eval_rc": None,
        "bench_json": None,
        "PSNR": None,
        "SSIM": None,
        "LPIPS": None,
        "FPS": None,
        "PLY_MB": None,
        "num_gaussians": None,
        "skipped": False,
        "skip_reason": None,
    }

    # sanity checks
    if not iter_dir.exists():
        info["skipped"] = True
        info["skip_reason"] = f"missing iter_dir: {iter_dir}"
        return info
    if not ply.exists():
        info["skipped"] = True
        info["skip_reason"] = f"missing ply: {ply}"
        return info
    if not npz_in.exists():
        info["skipped"] = True
        info["skip_reason"] = f"missing npz_in: {npz_in}"
        return info
    if not source_path.exists():
        info["skipped"] = True
        info["skip_reason"] = f"missing source_path: {source_path}"
        return info

    print("\n" + "=" * 80)
    print(f"[SCENE] {scene}")
    print(f"  model_path : {model_path}")
    print(f"  source_path: {source_path}")
    print(f"  npz_in     : {npz_in}")
    print(f"  npz_out    : {npz_out}")
    print("=" * 80)

    finetune_log = log_dir / f"ft_{scene}.log"
    eval_log = log_dir / f"eval_{scene}.log"

    # commands
    finetune_cmd = [
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

    eval_cmd = [
        sys.executable, str(repo_root / "gs_decode.py"),
        "--model_path", str(model_path),
        "--tag", tag_out,
        "--bench", str(bench_script),
        "-s", str(source_path),
        "--iteration", str(it),
        "--split", split,
        "--n_views", str(n_views),
        "--sh_degree", str(sh_degree),
    ]

    if dry_run:
        print("[DRY RUN] finetune cmd:\n", " ".join(finetune_cmd))
        print("[DRY RUN] eval cmd:\n", " ".join(eval_cmd))
        info["finetune_rc"] = 0
        info["eval_rc"] = 0
        return info

    # 1) finetune
    print("[1/2] Finetune codebook ...")
    rc1 = tee_run(finetune_cmd, finetune_log, cwd=repo_root)
    info["finetune_rc"] = rc1
    if rc1 != 0:
        print(f"[ERROR] finetune failed for {scene}, rc={rc1}. See {finetune_log}")
        return info

    # 2) decode + eval
    print("[2/2] Decode & benchmark ...")
    rc2 = tee_run(eval_cmd, eval_log, cwd=repo_root)
    info["eval_rc"] = rc2
    if rc2 != 0:
        print(f"[ERROR] eval failed for {scene}, rc={rc2}. See {eval_log}")
        return info

    # read bench json
    bj = find_bench_json(iter_dir, tag_out)
    if bj and bj.exists():
        info["bench_json"] = str(bj)
        try:
            data = json.loads(bj.read_text(encoding="utf-8"))
            for k in ["PSNR", "SSIM", "LPIPS", "FPS", "PLY_MB", "num_gaussians"]:
                if k in data:
                    info[k] = data[k]
        except Exception as e:
            print(f"[WARN] failed to parse bench json {bj}: {e}")

    return info

def write_summary_csv(rows: list[dict], out_csv: Path):
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "scene", "model_path", "source_path",
        "npz_in", "npz_out",
        "finetune_rc", "eval_rc",
        "PSNR", "SSIM", "LPIPS", "FPS", "PLY_MB", "num_gaussians",
        "bench_json", "skipped", "skip_reason"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in keys})

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

    ap.add_argument("--bench_script", default="/root/autodl-tmp/gaussian-splatting_feature-pruning/benchmark_fps.py")

    ap.add_argument("--log_dir", default=None, help="default: {output_root}/_vqft_logs")
    ap.add_argument("--summary_csv", default=None, help="default: {output_root}/_vqft_logs/summary.csv")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    path_360 = Path(args.path_360)
    path_tandt = Path(args.path_tandt)
    path_db = Path(args.path_db)
    bench_script = Path(args.bench_script)

    log_dir = Path(args.log_dir) if args.log_dir else (output_root / "_vqft_logs")
    summary_csv = Path(args.summary_csv) if args.summary_csv else (log_dir / "summary.csv")

    # tasks（按你给的）
    tasks = [
        ("MipNeRF360", path_360, ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]),
        ("TandT", path_tandt, ["train", "truck"]),
        ("DeepBlending", path_db, ["drjohnson", "playroom"]),
    ]

    all_rows = []
    for ds_name, ds_root, scenes in tasks:
        print(f"\n### DATASET: {ds_name}  root={ds_root}")
        for scene in scenes:
            row = run_scene(
                scene=scene,
                dataset_root=ds_root,
                output_root=output_root,
                it=args.iteration,
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
                bench_script=bench_script,
                repo_root=repo_root,
                log_dir=log_dir,
                dry_run=args.dry_run,
            )
            if row.get("skipped"):
                print(f"[SKIP] {scene}: {row.get('skip_reason')}")
            all_rows.append(row)

    # write summary
    write_summary_csv(all_rows, summary_csv)
    print(f"\n== DONE ==\nLogs: {log_dir}\nSummary CSV: {summary_csv}")

if __name__ == "__main__":
    main()
