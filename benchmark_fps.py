# benchmark_single_fixed.py
import os
import json
import time
import torch
import sys
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render

# ---- 引入指标计算模块 ----
from utils.image_utils import psnr as official_psnr
from utils.loss_utils import ssim as official_ssim
from lpipsPyTorch import lpips as official_lpips


def get_ply_mb(model_path: str, iteration: int):
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if os.path.exists(ply_path):
        return os.path.getsize(ply_path) / (1024 * 1024)
    return 0.0


def main():
    parser = ArgumentParser(description="Benchmark ALL metrics for a SINGLE scene")

    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", type=int, default=35000)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--n_views", type=int, default=-1, help="Test N views (default -1 = all)")
    parser.add_argument("--warmup", type=int, default=5, help="FPS warmup frames")
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["vgg", "alex"])
    parser.add_argument("--out_json", type=str, default=None, help="Save result to this json file")

    # 这一步会加载 cfg_args，可能会覆盖命令行参数
    args = get_combined_args(parser)

    # 【核心修复】强制开启 eval 模式，确保 Colmap 加载器划分测试集
    args.eval = True
    print(f"[INFO] Force setting args.eval = True to load test cameras.")

    print(f"[1/1] Evaluating Scene: {args.model_path}")

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    # 初始化模型
    gaussians = GaussianModel(dataset.sh_degree)

    # 加载场景
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    # 获取相机
    if args.split == "train":
        cameras = scene.getTrainCameras()
    else:
        cameras = scene.getTestCameras()

    print(f"Total cameras in {args.split} set: {len(cameras)}")

    if len(cameras) == 0:
        print("[ERROR] No cameras found! Check your dataset path and split.")
        return

    if args.n_views > 0:
        cameras = cameras[:args.n_views]
        print(f"Benchmarking first {len(cameras)} views...")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    psnr_list, ssim_list, lpips_list, fps_list = [], [], [], []

    torch.cuda.reset_peak_memory_stats()

    for idx, view in enumerate(cameras):
        torch.cuda.synchronize()
        start_t = time.time()

        # 强制关闭 exposure
        render_pkg = render(view, gaussians, pipe, background, use_trained_exp=False)
        render_result = render_pkg["render"]

        torch.cuda.synchronize()
        end_t = time.time()

        if idx >= args.warmup:
            fps_list.append(1.0 / (end_t - start_t))

        gt = view.original_image[0:3, :, :].cuda()

        p = official_psnr(render_result, gt).mean().float().item()
        s = official_ssim(render_result, gt).mean().float().item()
        l = official_lpips(render_result, gt, net_type=args.lpips_net).mean().float().item()

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)

        print(f"View {idx}: PSNR={p:.2f}, SSIM={s:.3f}, LPIPS={l:.3f}")

    mean_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    mean_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0
    mean_lpips = sum(lpips_list) / len(lpips_list) if lpips_list else 0
    mean_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    ply_mb = get_ply_mb(args.model_path, args.iteration)

    result = {
        "scene": scene.model_path,
        "iteration": args.iteration,
        "n_views": len(cameras),
        "PSNR": mean_psnr,
        "SSIM": mean_ssim,
        "LPIPS": mean_lpips,
        "FPS": mean_fps,
        "PeakMemMB": peak_mem,
        "PLY_MB": ply_mb,
        "num_gaussians": gaussians.get_xyz.shape[0]
    }

    print("\n==== EVALUATION RESULTS ====")
    print(json.dumps(result, indent=2))

    if hasattr(args, 'out_json') and args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {args.out_json}")


if __name__ == "__main__":
    main()