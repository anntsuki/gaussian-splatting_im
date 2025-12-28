# prune_importance.py
import os
import shutil
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
import torch

def _slice_if_compatible(gaussians, name, keep):
    if not hasattr(gaussians, name):
        return
    t = getattr(gaussians, name)
    if not isinstance(t, torch.Tensor):
        return
    # 空张量或长度不等于 N：跳过（否则就会你现在这种 IndexError）
    if t.numel() == 0:
        return
    if t.shape[0] != keep.shape[0]:
        return
    setattr(gaussians, name, t[keep])

def _slice_param(param, keep):
    # 保持为 nn.Parameter（更兼容 save_ply/后续可能的渲染）
    return torch.nn.Parameter(param[keep].contiguous(), requires_grad=False)

def prune_gaussians_inplace(gaussians, prune_mask):
    keep = (~prune_mask).to(gaussians._xyz.device)

    # 核心 3DGS 参数（这些一定要裁剪）
    gaussians._xyz = _slice_param(gaussians._xyz, keep)
    gaussians._features_dc = _slice_param(gaussians._features_dc, keep)
    gaussians._features_rest = _slice_param(gaussians._features_rest, keep)
    gaussians._opacity = _slice_param(gaussians._opacity, keep)
    gaussians._scaling = _slice_param(gaussians._scaling, keep)
    gaussians._rotation = _slice_param(gaussians._rotation, keep)

    # 这些是训练/统计缓存：只有在“非空且长度匹配 N”时才裁
    _slice_if_compatible(gaussians, "max_radii2D", keep)
    _slice_if_compatible(gaussians, "xyz_gradient_accum", keep)
    _slice_if_compatible(gaussians, "denom", keep)

@torch.no_grad()
def main():
    parser = ArgumentParser("3DGS post-pruning (opacity / contribution)")

    # 注意：ModelParams 用 sentinel=True 才能更好地配合 cfg_args 合并
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", type=int, default=-1, help="Which trained iteration to load (-1 = latest)")
    parser.add_argument("--out_model_path", type=str, required=True, help="Output folder for pruned model")

    parser.add_argument("--prune_ratio", type=float, default=0.5, help="Prune ratio, e.g. 0.5 = remove 50%")
    parser.add_argument(
        "--score_mode",
        type=str,
        default="opacity_area",
        choices=["opacity", "opacity_area", "opacity_area_vis"],
        help="How to score each gaussian"
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max_cams", type=int, default=-1, help="Use only first K cameras (-1 = all)")

    args = get_combined_args(parser)
    if not hasattr(args, "depths"):
        args.depths = ""
    if not hasattr(args, "train_test_exp"):
        args.train_test_exp = False
    if not hasattr(args, "data_device"):
        args.data_device = "cuda"

    # 1) 载入训练好的模型（会从 point_cloud/iteration_x/point_cloud.ply 读取）
    # Scene 的 load_iteration 逻辑就是这么干的:contentReference[oaicite:3]{index=3}
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    # 为了能用 prune_points + capture（里面需要 optimizer state），先做一次 training_setup
    # gaussians.training_setup(opt)

    # 2) 准备相机列表
    cams = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if args.max_cams is not None and args.max_cams > 0:
        cams = cams[: args.max_cams]

    # 背景色（官方训练也是这么设的）:contentReference[oaicite:4]{index=4}
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 3) 统计贡献：visibility_filter + radii 来自 render_pkg:contentReference[oaicite:5]{index=5}
    N = gaussians.get_xyz.shape[0]
    area_sum = torch.zeros((N,), device="cuda")
    vis_cnt = torch.zeros((N,), dtype=torch.int32, device="cuda")

    for cam in cams:
        pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
        radii = pkg["radii"]  # (N,)
        vis = pkg["visibility_filter"]  # indices of visible gaussians:contentReference[oaicite:6]{index=6}
        idx = vis.squeeze(-1)

        vis_cnt[idx] += 1
        area_sum[idx] += radii[idx] * radii[idx]  # 用 r^2 近似屏幕投影面积权重

    opacity = gaussians.get_opacity.squeeze()  # sigmoid后的 opacity:contentReference[oaicite:7]{index=7}

    if args.score_mode == "opacity":
        score = opacity
    elif args.score_mode == "opacity_area":
        score = opacity * area_sum
    else:  # opacity_area_vis
        score = opacity * area_sum * vis_cnt.float()

    # 4) 按目标压缩率剪枝
    keep_k = max(1, int(N * (1.0 - args.prune_ratio)))
    keep_idx = torch.topk(score, k=keep_k, largest=True).indices

    prune_mask = torch.ones((N,), dtype=torch.bool, device="cuda")
    prune_mask[keep_idx] = False

    # 官方也是 prune_mask -> prune_points 这个模式:contentReference[oaicite:8]{index=8}
    prune_gaussians_inplace(gaussians, prune_mask)

    # 5) 保存为一个“可继续训练/可评测”的新输出目录
    out_root = args.out_model_path
    os.makedirs(out_root, exist_ok=True)

    # 复制 cfg_args（方便 render.py/metrics.py 继续用 get_combined_args 读配置）:contentReference[oaicite:9]{index=9}
    copy_if_exists(os.path.join(dataset.model_path, "cfg_args"), os.path.join(out_root, "cfg_args"))
    copy_if_exists(os.path.join(dataset.model_path, "cameras.json"), os.path.join(out_root, "cameras.json"))
    copy_if_exists(os.path.join(dataset.model_path, "exposure.json"), os.path.join(out_root, "exposure.json"))
    copy_if_exists(os.path.join(dataset.model_path, "input.ply"), os.path.join(out_root, "input.ply"))

    # 保存 pruned ply（放到 iteration_0 下，方便你区分）
    pc_dir = os.path.join(out_root, "point_cloud", "iteration_0")
    os.makedirs(pc_dir, exist_ok=True)
    out_ply = os.path.join(pc_dir, "point_cloud.ply")
    gaussians.save_ply(out_ply)

    # # 额外：保存一个 checkpoint，给 train.py 的 --start_checkpoint 直接用:contentReference[oaicite:10]{index=10}
    # out_ckpt = os.path.join(out_root, "chkpnt_pruned.pth")
    # torch.save((gaussians.capture(), 0), out_ckpt)

    print(f"[Done] N={N} -> {gaussians.get_xyz.shape[0]} kept ({keep_k})")
    print(f"[Saved] ply:  {out_ply}")
    # print(f"[Saved] ckpt: {out_ckpt}")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
