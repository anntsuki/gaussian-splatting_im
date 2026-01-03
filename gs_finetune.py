import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import numpy as np
from tqdm import tqdm
from random import randint
from math import exp
from torch.autograd import Variable

# 引入 3DGS 核心模块
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss


# ================= SSIM 实现 =================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ============================================================

def quant_minmax(x_float, bits=8):
    # 重新计算 min/max 并量化
    mn = x_float.min(dim=0)[0]
    mx = x_float.max(dim=0)[0]
    span = mx - mn
    span[span < 1e-9] = 1.0
    qmax = (1 << bits) - 1

    x_norm = (x_float - mn) / span
    x_q = torch.round(x_norm * qmax).clamp(0, qmax).to(
        torch.uint8 if bits == 8 else torch.int16)  # changed to torch type
    return x_q.cpu().numpy(), mn.cpu().numpy(), mx.cpu().numpy()


def quant_quat(rot_float, bits=16):
    # 归一化
    rot_float = F.normalize(rot_float, dim=1)
    if bits == 16:
        q = torch.round(torch.clamp(rot_float, -1, 1) * 32767.0).to(torch.int16)
    else:
        q = torch.round(torch.clamp(rot_float, -1, 1) * 127.0).to(torch.int8)
    return q.cpu().numpy()


def finetune(dataset, opt, pipe, args):
    iter_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}")
    npz_path = os.path.join(iter_dir, f"point_cloud.{args.tag}.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Compressed model not found at {npz_path}")

    print(f"[FT] Loading compressed model from {npz_path}...")
    data = np.load(npz_path)
    device = "cuda"

    # 1. 解压并设置为可训练 (Requires Grad = True)
    # Codebooks
    cb_r = nn.Parameter(torch.from_numpy(data["cb_r"]).to(device).float().requires_grad_(True))
    cb_g = nn.Parameter(torch.from_numpy(data["cb_g"]).to(device).float().requires_grad_(True))
    cb_b = nn.Parameter(torch.from_numpy(data["cb_b"]).to(device).float().requires_grad_(True))

    idx_r = torch.from_numpy(data["idx_r"].astype(np.int64)).to(device)
    idx_g = torch.from_numpy(data["idx_g"].astype(np.int64)).to(device)
    idx_b = torch.from_numpy(data["idx_b"].astype(np.int64)).to(device)

    # XYZ (解封！允许微调)
    xyz = nn.Parameter(torch.from_numpy(data["pos16"].astype(np.float32)).to(device).requires_grad_(True))

    # Rotation (解封！允许微调)
    rot_q = torch.from_numpy(data["rot_q"]).to(device)
    rot_bits = int(data["rot_bits"])
    norm_factor = 32767.0 if rot_bits == 16 else 127.0
    rot_val = (rot_q.float() / norm_factor)
    rot_val = F.normalize(rot_val, dim=1)
    rot = nn.Parameter(rot_val.requires_grad_(True))

    # Scale (解封！允许微调)
    scale_q = torch.from_numpy(data["sc_q"]).to(device)
    scale_mn = torch.tensor(data["sc_mn"]).to(device)
    scale_mx = torch.tensor(data["sc_mx"]).to(device)
    scale_val = scale_mn + (scale_mx - scale_mn) * (scale_q.float() / 255.0)
    scale = nn.Parameter(scale_val.requires_grad_(True))

    # Opacity (解封！允许微调)
    op_q = torch.from_numpy(data["op_q"]).to(device)
    op_mn = torch.tensor(data["op_mn"]).to(device)
    op_mx = torch.tensor(data["op_mx"]).to(device)
    opacity_val = op_mn + (op_mx - op_mn) * (op_q.float() / 255.0)
    opacity = nn.Parameter(opacity_val.requires_grad_(True))

    print(f"[FT] Loaded {xyz.shape[0]} Gaussians. ALL parameters unlocked for finetuning.")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=True)

    # 2. 设置优化器 (为几何参数设置较小的学习率，防止飞掉)
    optimizer = optim.Adam([
        {'params': [cb_r, cb_g, cb_b], 'lr': 0.01, "name": "codebook"},
        {'params': [opacity], 'lr': 0.02, "name": "opacity"},
        {'params': [xyz], 'lr': 0.00016 * scene.cameras_extent, "name": "xyz"},  # 标准3DGS的学习率
        {'params': [rot], 'lr': 0.001, "name": "rotation"},
        {'params': [scale], 'lr': 0.005, "name": "scaling"}
    ], lr=0.0)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    pbar = tqdm(range(args.finetune_iters), desc="Full Finetuning")

    for iteration in pbar:
        try:
            viewpoint_cam = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras()) - 1)]
        except:
            viewpoint_cam = scene.getTrainCameras()[0]

        f_r = torch.index_select(cb_r, 0, idx_r)
        f_g = torch.index_select(cb_g, 0, idx_g)
        f_b = torch.index_select(cb_b, 0, idx_b)

        # DC: [N, 1, 3]
        dc = torch.stack([f_r[:, 0], f_g[:, 0], f_b[:, 0]], dim=1).unsqueeze(1)

        # Rest: [N, Coeffs, 3]
        if f_r.shape[1] > 1:
            rest_r = f_r[:, 1:]
            rest_g = f_g[:, 1:]
            rest_b = f_b[:, 1:]
            features_rest = torch.stack([rest_r, rest_g, rest_b], dim=-1)
        else:
            features_rest = torch.zeros((xyz.shape[0], 0, 3), device=device)

        gaussians._xyz = xyz
        gaussians._features_dc = dc
        gaussians._features_rest = features_rest
        gaussians._opacity = opacity
        gaussians._scaling = scale
        gaussians._rotation = rot

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                    1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iteration % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})

    print(f"[FT] Saving re-quantized model to {args.save_path}...")

    # 3. 重新量化并保存
    # Codebook (FP16)
    new_cb_r = cb_r.detach().cpu().numpy().astype(np.float16)
    new_cb_g = cb_g.detach().cpu().numpy().astype(np.float16)
    new_cb_b = cb_b.detach().cpu().numpy().astype(np.float16)

    # XYZ (FP16) - 只有 FP16 才能保持体积小
    new_pos16 = xyz.detach().cpu().numpy().astype(np.float16)

    # Opacity (Re-quantize to 8bit)
    new_op_q, op_mn_new, op_mx_new = quant_minmax(opacity.detach(), bits=8)

    # Scale (Re-quantize to 8bit)
    new_sc_q, sc_mn_new, sc_mx_new = quant_minmax(scale.detach(), bits=8)

    # Rotation (Re-quantize to bits)
    new_rot_q = quant_quat(rot.detach(), bits=data["rot_bits"])

    np.savez_compressed(
        args.save_path,
        degree=data["degree"],
        K=data["K"],
        morton_used=data["morton_used"],
        idx_r=data["idx_r"], idx_g=data["idx_g"], idx_b=data["idx_b"],
        # 更新所有几何参数
        pos16=new_pos16,
        nrm16=data["nrm16"],  # Normal 通常不微调
        sc_q=new_sc_q, sc_mn=sc_mn_new, sc_mx=sc_mx_new, sc_bits=data["sc_bits"],
        rot_q=new_rot_q, rot_bits=data["rot_bits"],
        op_q=new_op_q, op_mn=op_mn_new, op_mx=op_mx_new, op_bits=data["op_bits"],
        cb_r=new_cb_r, cb_g=new_cb_g, cb_b=new_cb_b,
    )
    print("[FT] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--finetune_iters", type=int, default=5000)  # 默认加到 5000
    parser.add_argument("--iteration", type=int, default=30000)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args()

    if not args.model_path:
        parser.error("argument --model_path/-m is required")
    if not args.source_path:
        parser.error("argument --source_path/-s is required")

    if args.save_path is None:
        args.save_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}",
                                      f"point_cloud.{args.tag}.finetuned.npz")

    finetune(lp.extract(args), op.extract(args), pp.extract(args), args)