# benchmark_single_scene.py
import os
import sys
import json
import time
import torch
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


@torch.no_grad()
def evaluate_single_scene(model_path, source_path, iteration, split, n_views, warmup, lpips_net, out_json):
    # 1. 准备参数
    # 我们需要手动构建 args 对象，或者修改 ArgumentParser
    # 这里为了简单，我们直接用 parser 解析，假设用户会传入对应的参数
    pass


def main():
    parser = ArgumentParser(description="Benchmark ALL metrics for a SINGLE scene")

    # 标准 3DGS 参数解析
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    # 这里的 -m 是具体的某个场景路径，例如 .../bicycle
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--n_views", type=int, default=-1, help="Test N views (default -1 = all)")
    parser.add_argument("--warmup", type=int, default=5, help="FPS warmup frames")
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["vgg", "alex"])
    parser.add_argument("--out_json", type=str, default=None, help="Save result to this json file")

    args = get_combined_args(parser)

    print(f"[1/1] Evaluating Scene: {args.model_path}")

    # 1. 准备数据集和模型
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    # 这里的 source_path 如果没传，会尝试从 cfg_args 读取，Argument解析器会自动处理

    gaussians = GaussianModel(dataset.sh_degree)

    # 加载场景
    # load_iteration 决定加载哪个 ply
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    # 2. 确定测试相机列表
    if args.split == "train":
        cameras = scene.getTrainCameras()
    else:
        cameras = scene.getTestCameras()

    print(f"Total cameras in {args.split} set: {len(cameras)}")

    if args.n_views > 0:
        cameras = cameras[:args.n_views]
        print(f"Benchmarking first {len(cameras)} views...")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 3. 开始循环测试
    psnr_list = []
    ssim_list = []
    lpips_list = []
    fps_list = []

    # 显存统计
    torch.cuda.reset_peak_memory_stats()

    for idx, view in enumerate(cameras):
        # 渲染
        torch.cuda.synchronize()
        start_t = time.time()

        # 强制使用训练时的 exposure 设置（通常是 False 或 True，视你的训练而定）
        # 这里用 dataset.train_test_exp 来控制，如果之前脚本禁用了，这里也应该保持一致
        # 对于 Co-adaptation 后的模型，建议显式 False，或者让它读 cfg_args
        render_result = render(view, gaussians, pipe, background)["render"]

        torch.cuda.synchronize()
        end_t = time.time()

        # 记录 FPS (跳过 warmup)
        if idx >= args.warmup:
            fps_list.append(1.0 / (end_t - start_t))

        # 计算指标
        gt = view.original_image[0:3, :, :].cuda()

        # PSNR
        p = official_psnr(render_result, gt).mean().float().item()
        psnr_list.append(p)

        # SSIM
        s = official_ssim(render_result, gt).mean().float().item()
        ssim_list.append(s)

        # LPIPS
        l = official_lpips(render_result, gt, net_type=args.lpips_net).mean().float().item()
        lpips_list.append(l)

        print(f"View {idx}: PSNR={p:.2f}, SSIM={s:.3f}, LPIPS={l:.3f}")

    # 4. 汇总结果
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

    # 打印到控制台
    print("\n==== EVALUATION RESULTS ====")
    print(json.dumps(result, indent=2))

    # 保存到文件
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {args.out_json}")


if __name__ == "__main__":
    main()