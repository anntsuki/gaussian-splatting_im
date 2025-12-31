# benchmark_all_metrics.py
import os
import sys
import json
import csv
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render

# ---- Use OFFICIAL metric implementations (same as metrics.py you sent) ----
from utils.image_utils import psnr as official_psnr
from utils.loss_utils import ssim as official_ssim
from lpipsPyTorch import lpips as official_lpips


def discover_model_paths(models_root: str):
    """Discover per-scene model_path under models_root (one-level)."""
    paths = []
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"models_root 不存在或不是目录：{models_root}")

    for name in sorted(os.listdir(models_root)):
        p = os.path.join(models_root, name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "cfg_args")) or os.path.exists(os.path.join(p, "point_cloud")):
            paths.append(p)
    return paths


def get_ply_mb(model_path: str, iteration: int):
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if os.path.exists(ply_path):
        return os.path.getsize(ply_path) / (1024**2)
    return None


@torch.no_grad()
def eval_one_model(
    model_path: str,
    iteration: int,
    split: str,
    n_views: int,
    warmup: int,
    lpips_net: str = "vgg",  # keep same default as official metrics.py
):
    """
    Evaluate one model_path and output:
    PSNR/SSIM/LPIPS using official functions
    FPS/PeakMem using CUDA events (render-only timing)
    #Gaussians and PLY size
    """

    # ---- Reuse get_combined_args so it loads cfg_args inside model_path
    argv_backup = sys.argv
    try:
        parser = ArgumentParser()
        lp = ModelParams(parser, sentinel=True)
        pp = PipelineParams(parser)
        op = OptimizationParams(parser)

        parser.add_argument("--iteration", type=int, default=iteration)
        parser.add_argument("--split", type=str, default=split, choices=["test", "train"])
        parser.add_argument("--n_views", type=int, default=n_views)
        parser.add_argument("--warmup", type=int, default=warmup)

        sys.argv = [
            argv_backup[0],
            "-m", model_path,
            "--iteration", str(iteration),
            "--split", split,
            "--n_views", str(n_views),
            "--warmup", str(warmup),
        ]
        args = get_combined_args(parser)
    finally:
        sys.argv = argv_backup

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    cams = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    if len(cams) == 0:
        raise RuntimeError(f"{model_path} split={args.split} 没有相机可用")

    cams = cams[: min(len(cams), args.n_views)]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---------------- FPS / PeakMem (render-only)
    for i in range(min(args.warmup, len(cams))):
        _ = render(cams[i], gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)["render"]
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for cam in cams:
        start.record()
        _ = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)["render"]
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_ms = float(sum(times_ms) / len(times_ms))
    fps = 1000.0 / avg_ms
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # ---------------- Quality metrics (OFFICIAL-style)
    # IMPORTANT: official metrics.py reads PNG -> to_tensor -> [1,3,H,W] in [0,1]
    # Here we mimic that interface: [1,3,H,W] in [0,1] WITHOUT extra normalization/masking.
    psnrs, ssims, lpipss = [], [], []
    for cam in cams:
        pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
        img = pkg["render"].clamp(0, 1)
        gt = cam.original_image.to("cuda").clamp(0, 1)

        img_b = img.unsqueeze(0)  # [1,3,H,W]
        gt_b = gt.unsqueeze(0)

        p = official_psnr(img_b, gt_b)
        s = official_ssim(img_b, gt_b)
        l = official_lpips(img_b, gt_b, net_type=lpips_net)

        # official funcs may return tensors; robustly reduce to scalar
        psnrs.append(float(torch.mean(p).item()) if torch.is_tensor(p) else float(p))
        ssims.append(float(torch.mean(s).item()) if torch.is_tensor(s) else float(s))
        lpipss.append(float(torch.mean(l).item()) if torch.is_tensor(l) else float(l))

    psnr_avg = float(sum(psnrs) / len(psnrs))
    ssim_avg = float(sum(ssims) / len(ssims))
    lpips_avg = float(sum(lpipss) / len(lpipss))

    # other stats
    ply_mb = get_ply_mb(model_path, args.iteration)
    n_pts = int(gaussians.get_xyz.shape[0])

    out = {
        "scene": os.path.basename(model_path.rstrip("/")),
        "model_path": model_path,
        "iteration": int(args.iteration),
        "split": args.split,
        "n_views": len(cams),

        "PSNR": psnr_avg,
        "SSIM": ssim_avg,
        "LPIPS": lpips_avg,

        "avg_ms": avg_ms,
        "FPS": fps,
        "PeakMemMB": peak_mem_mb,

        "num_gaussians": n_pts,
        "PLY_MB": ply_mb,
        "lpips_net": lpips_net,
    }
    return out


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--models_root", type=str, required=True,
                        help="包含多个场景 model_path 的根目录（例如 eval_line2）")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--n_views", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["vgg", "alex"])
    parser.add_argument("--out_json", type=str, default="all_metrics.json")
    parser.add_argument("--out_csv", type=str, default="all_metrics.csv")
    args = parser.parse_args()

    torch.cuda.set_device(0)

    model_paths = discover_model_paths(args.models_root)
    if len(model_paths) == 0:
        raise RuntimeError(f"在 {args.models_root} 下未发现任何场景目录（需要含 cfg_args 或 point_cloud）")

    results = []
    for i, mp in enumerate(model_paths):
        print(f"\n[{i+1}/{len(model_paths)}] Evaluating: {mp}")
        try:
            torch.cuda.empty_cache()
            out = eval_one_model(
                model_path=mp,
                iteration=args.iteration,
                split=args.split,
                n_views=args.n_views,
                warmup=args.warmup,
                lpips_net=args.lpips_net,
            )
            results.append(out)
            print(json.dumps(out, indent=2))
        except Exception as e:
            print(f"!! Failed on {mp}: {e}")

    # summary
    def mean_of(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return (sum(vals) / len(vals)) if len(vals) > 0 else None

    summary = {
        "models_root": args.models_root,
        "num_scenes_ok": len(results),
        "lpips_net": args.lpips_net,
        "mean_PSNR": mean_of("PSNR"),
        "mean_SSIM": mean_of("SSIM"),
        "mean_LPIPS": mean_of("LPIPS"),
        "mean_FPS": mean_of("FPS"),
        "mean_PeakMemMB": mean_of("PeakMemMB"),
        "mean_num_gaussians": mean_of("num_gaussians"),
        "mean_PLY_MB": mean_of("PLY_MB"),
    }

    payload = {"summary": summary, "results": results}

    out_json_path = os.path.join(args.models_root, args.out_json)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    out_csv_path = os.path.join(args.models_root, args.out_csv)
    fieldnames = [
        "scene", "iteration", "split", "n_views",
        "PSNR", "SSIM", "LPIPS",
        "FPS", "avg_ms", "PeakMemMB",
        "num_gaussians", "PLY_MB",
        "lpips_net",
        "model_path",
    ]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fieldnames})

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved:\n  JSON: {out_json_path}\n  CSV : {out_csv_path}")


if __name__ == "__main__":
    main()
