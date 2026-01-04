#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import json
import csv
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any


def tee_run(cmd: List[str], log_path: Path, cwd: Optional[Path] = None) -> int:
    """Run command and tee stdout to console + log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=== CMD ===\n{}\n\n".format(" ".join(cmd)))
        f.write("=== START === {}\n\n".format(datetime.now().isoformat(timespec="seconds")))
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
            print(line, end="")
            f.write(line)

        p.wait()
        f.write("\n=== END === {}  rc={} ===\n".format(datetime.now().isoformat(timespec="seconds"), p.returncode))
        f.flush()
        return int(p.returncode)


def find_bench_json(iter_dir: Path, tag: str) -> Optional[Path]:
    """
    优先找 bench_{tag}.json
    否则在 iteration 目录下找任意 bench_*.json 包含 tag 的
    """
    p = iter_dir / ("bench_{}.json".format(tag))
    if p.exists():
        return p

    for cand in sorted(iter_dir.glob("bench_*.json")):
        if tag in cand.name:
            return cand
    return None


def write_summary_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "scene", "dataset",
        "model_path", "source_path",
        "npz_in", "npz_out",
        "finetune_rc", "decode_eval_rc",
        "PSNR", "SSIM", "LPIPS", "FPS", "PLY_MB", "num_gaussians",
        "bench_json", "skipped", "skip_reason",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in keys})


def run_scene(
    scene: str,
    dataset: str,
    dataset_root: Path,
    output_root: Path,
    it: int,
    tag_in: str,
    tag_out: str,
    steps: int,
    lr: float,
    batch: int,
    reassign_every: int,
    device: str,
    split: str,
    n_views: int,
    sh_degree: int,
    bench_script: Optional[Path],
    repo_root: Path,
    log_dir: Path,
) -> Dict[str, Any]:

    model_path = output_root / scene
    iter_dir = model_path / "point_cloud" / ("iteration_{}".format(it))

    ply = iter_dir / "point_cloud.ply"
    npz_in = iter_dir / ("point_cloud.{}.npz".format(tag_in))
    npz_out = iter_dir / ("point_cloud.{}.npz".format(tag_out))
    source_path = dataset_root / scene

    row: Dict[str, Any] = {
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
        "bench_json": None,
        "skipped": False,
        "skip_reason": None,
    }

    # sanity checks
    if not iter_dir.exists():
        row["skipped"] = True
        row["skip_reason"] = "missing iter_dir: {}".format(iter_dir)
        return row
    if not ply.exists():
        row["skipped"] = True
        row["skip_reason"] = "missing ply: {}".format(ply)
        return row
    if not npz_in.exists():
        row["skipped"] = True
        row["skip_reason"] = "missing npz_in: {}".format(npz_in)
        return row
    if not source_path.exists():
        row["skipped"] = True
        row["skip_reason"] = "missing source_path: {}".format(source_path)
        return row

    print("\n" + "=" * 90)
    print("[{}] scene={}".format(dataset, scene))
    print("  model_path : {}".format(model_path))
    print("  source_path: {}".format(source_path))
    print("  npz_in     : {}".format(npz_in))
    print("  npz_out    : {}".format(npz_out))
    print("=" * 90)

    ft_log = log_dir / ("ft_{}.log".format(scene))
    eval_log = log_dir / ("eval_{}.log".format(scene))

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
        print("[ERROR] finetune failed: {}  rc={}  log={}".format(scene, rc1, ft_log))
        return row

    # 2) decode + eval (gs_decode.py 自带 benchmark 输出)
    eval_cmd = [
        sys.executable, str(repo_root / "gs_decode.py"),
        "--model_path", str(model_path),
        "--tag", tag_out,
        "-s", str(source_path),
        "--iteration", str(it),
        "--split", split,
        "--n_views", str(n_views),
        "--sh_degree", str(sh_degree),
    ]
    # 如果你平时习惯显式传 bench，就加上
    if bench_script is not None:
        eval_cmd += ["--bench", str(bench_script)]

    print("[2/2] decode + eval ...")
    rc2 = tee_run(eval_cmd, eval_log, cwd=repo_root)
    row["decode_eval_rc"] = rc2
    if rc2 != 0:
        print("[ERROR] decode/eval failed: {}  rc={}  log={}".format(scene, rc2, eval_log))
        return row

    # parse bench json
    bj = find_bench_json(iter_dir, tag_out)
    if bj is not None and bj.exists():
        row["bench_json"] = str(bj)
        try:
            data = json.loads(bj.read_text(encoding="utf-8"))
            for k in ["PSNR", "SSIM", "LPIPS", "FPS", "PLY_MB", "num_gaussians"]:
                if k in data:
                    row[k] = data[k]
        except Exception as e:
            print("[WARN] failed to parse bench json {}: {}".format(bj, e))

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

    # 你说 decode 直接出测评；这里 bench_script 默认填你 repo 里的 benchmark_fps.py 路径（你也可以传空禁用）
    ap.add_argument("--bench_script", default="/root/autodl-tmp/gaussian-splatting_feature-pruning/benchmark_fps.py")
    ap.add_argument("--no_bench_arg", action="store_true", help="不向 gs_decode.py 传 --bench 参数（用它内部默认）")

    ap.add_argument("--log_dir", default=None, help="default: {output_root}/_vqft_logs")
    ap.add_argument("--summary_csv", default=None, help="default: {log_dir}/summary.csv")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_root = Path(args.output_root)

    log_dir = Path(args.log_dir) if args.log_dir else (output_root / "_vqft_logs")
    summary_csv = Path(args.summary_csv) if args.summary_csv else (log_dir / "summary.csv")

    bench_script = None if args.no_bench_arg else Path(args.bench_script)

    tasks = [
        ("MipNeRF360", Path(args.path_360), ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]),
        ("TandT", Path(args.path_tandt), ["train", "truck"]),
        ("DeepBlending", Path(args.path_db), ["drjohnson", "playroom"]),
    ]

    rows: List[Dict[str, Any]] = []
    for dataset, ds_root, scenes in tasks:
        print("\n### DATASET: {}  root={}".format(dataset, ds_root))
        for scene in scenes:
            r = run_scene(
                scene=scene,
                dataset=dataset,
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
            )
            if r.get("skipped"):
                print("[SKIP] {}: {}".format(scene, r.get("skip_reason")))
            rows.append(r)

    write_summary_csv(rows, summary_csv)
    print("\n== DONE ==\nLogs: {}\nSummary CSV: {}".format(log_dir, summary_csv))


if __name__ == "__main__":
    main()
