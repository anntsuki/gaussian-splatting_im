# benchmark_all_metrics.py
import os
import sys
import json
import csv
import time
import math
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render


# -----------------------------
# Metrics (self-contained)
# -----------------------------
def psnr_torch(img: torch.Tensor, gt: torch.Tensor, eps: float = 1e-10) -> float:
    # img, gt: [3,H,W] in [0,1]
    mse = torch.mean((img - gt) ** 2)
    mse = torch.clamp(mse, min=eps)
    return float((-10.0 * torch.log10(mse)).item())


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.view(1, 1, -1)
    window_2d = window_1d.transpose(2, 1) @ window_1d  # [1,1,ws,ws]
    return window_2d


def ssim_torch(img: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    SSIM over RGB, average over channels.
    img, gt: [3,H,W] in [0,1]
    """
    # to [1,3,H,W]
    x = img.unsqueeze(0)
    y = gt.unsqueeze(0)

    device, dtype = x.device, x.dtype
    window = _gaussian_window(window_size, sigma, device, dtype)
    window = window.expand(3, 1, window_size, window_size)  # groups=3

    # compute statistics
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=3)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=3)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=3) - mu_xy

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return float(ssim_map.mean().item())


def make_lpips(net: str = "vgg"):
    try:
        import lpips
    except Exception as e:
        raise RuntimeError(
            "未找到 lpips 包。请先安装：pip install lpips\n"
            f"原始错误：{e}"
        )
    model = lpips.LPIPS(net=net)
    return model


@torch.no_grad()
def lpips_torch(lpips_model, img: torch.Tensor, gt: torch.Tensor) -> float:
    """
    img, gt: [3,H,W] in [0,1]
    LPIPS expects [-1,1], [1,3,H,W]
    """
    x = img.unsqueeze(0) * 2.0 - 1.0
    y = gt.unsqueeze(0) * 2.0 - 1.0
    v = lpips_model(x, y)
    return float(v.mean().item())


# -----------------------------
# Utils
# -----------------------------
def discover_model_paths(models_root: str):
    paths = []
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"models_root 不存在或不是目录：{models_root}")

    for name in sorted(os.listdir(models_root)):
        p = os.path.join(models_root, name)
        if not os.path.isdir(p):
            continue
        # 典型 output 目录会有 cfg_args 和 point_cloud
        if os.path.exists(os.path.join(p, "cfg_args")) or os.path.exists(os.path.join(p, "point_cloud")):
            paths.append(p)
    return paths


def get_ply_mb(model_path: str, iteration: int):
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if os.path.exists(ply_path):
        return os.path.getsize(ply_path) / (1024**2)
    return None


def apply_mask_and_background(img: torch.Tensor, gt: torch.Tensor, cam, background: torch.Tensor):
    """
    让 img/gt 在 mask 外填充背景，避免透明区域影响指标。
    """
    mask = None
    # 不同分支可能字段名不同，尽量兼容
    for key in ["gt_alpha_mask", "alpha_mask", "mask"]:
        if hasattr(cam, key):
            m = getattr(cam, key)
            if m is not None:
                mask = m
                break

    if mask is None:
        return img, gt

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # [1,H,W]
    if mask.shape[0] == 1:
        mask = mask.repeat(3, 1, 1)  # [3,H,W]

    mask = mask.to(img.device, dtype=img.dtype).clamp(0, 1)
    bg = background.view(3, 1, 1).to(img.dtype)

    img2 = img * mask + bg * (1 - mask)
    gt2 = gt * mask + bg * (1 - mask)
    return img2, gt2


@torch.no_grad()
def eval_one_model(model_path: str, iteration: int, split: str, n_views: int, warmup: int,
                   lpips_model=None, lpips_enabled=True):
    # 为了复用 get_combined_args（它会读取 model_path 下的 cfg_args），这里临时改 sys.argv
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

        sys.argv = [argv_backup[0], "-m", model_path,
                    "--iteration", str(iteration),
                    "--split", split,
                    "--n_views", str(n_views),
                    "--warmup", str(warmup)]
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

    # ---------------- FPS / PeakMem（只计 render）
    # warmup
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

    # ---------------- Quality metrics（再渲一次，避免把 metric 计入 FPS）
    psnrs, ssims, lpipss = [], [], []
    for cam in cams:
        pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
        img = pkg["render"].clamp(0, 1)
        gt = cam.original_image.to("cuda").clamp(0, 1)

        img, gt = apply_mask_and_background(img, gt, cam, background)

        psnrs.append(psnr_torch(img, gt))
        ssims.append(ssim_torch(img, gt))

        if lpips_enabled:
            if lpips_model is None:
                raise RuntimeError("lpips_enabled=True 但 lpips_model=None")
            lpipss.append(lpips_torch(lpips_model, img, gt))

    psnr_avg = float(sum(psnrs) / len(psnrs))
    ssim_avg = float(sum(ssims) / len(ssims))
    lpips_avg = float(sum(lpipss) / len(lpipss)) if lpips_enabled else None

    # others
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
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--out_json", type=str, default="all_metrics.json")
    parser.add_argument("--out_csv", type=str, default="all_metrics.csv")
    args = parser.parse_args()

    torch.cuda.set_device(0)

    model_paths = discover_model_paths(args.models_root)
    if len(model_paths) == 0:
        raise RuntimeError(f"在 {args.models_root} 下未发现任何场景目录（需要含 cfg_args 或 point_cloud）")

    lpips_model = None
    lpips_enabled = (not args.no_lpips)
    if lpips_enabled:
        lpips_model = make_lpips(net=args.lpips_net).cuda().eval()

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
                lpips_model=lpips_model,
                lpips_enabled=lpips_enabled,
            )
            results.append(out)
            print(json.dumps(out, indent=2))
        except Exception as e:
            print(f"!! Failed on {mp}: {e}")

    # 汇总均值（只对成功的场景）
    def mean_of(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return (sum(vals) / len(vals)) if len(vals) > 0 else None

    summary = {
        "models_root": args.models_root,
        "num_scenes_ok": len(results),
        "mean_PSNR": mean_of("PSNR"),
        "mean_SSIM": mean_of("SSIM"),
        "mean_LPIPS": mean_of("LPIPS"),
        "mean_FPS": mean_of("FPS"),
        "mean_PeakMemMB": mean_of("PeakMemMB"),
        "mean_num_gaussians": mean_of("num_gaussians"),
        "mean_PLY_MB": mean_of("PLY_MB"),
    }

    payload = {"summary": summary, "results": results}

    # 写 JSON
    out_json_path = os.path.join(args.models_root, args.out_json)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 写 CSV
    out_csv_path = os.path.join(args.models_root, args.out_csv)
    fieldnames = [
        "scene", "iteration", "split", "n_views",
        "PSNR", "SSIM", "LPIPS",
        "FPS", "avg_ms", "PeakMemMB",
        "num_gaussians", "PLY_MB",
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
