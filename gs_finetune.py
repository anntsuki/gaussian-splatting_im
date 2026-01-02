import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import numpy as np
from tqdm import tqdm
from random import randint

# 引入 3DGS 核心模块
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim


def finetune(dataset, opt, pipe, args):
    # 1. 路径设置
    iter_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}")
    npz_path = os.path.join(iter_dir, f"point_cloud.{args.tag}.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Compressed model not found at {npz_path}")

    print(f"[FT] Loading compressed model from {npz_path}...")
    data = np.load(npz_path)

    # 2. 准备数据 & Codebooks (转为 Tensor 并设为可导)
    device = "cuda"

    # --- 加载 Codebooks (核心优化对象) ---
    # 自动识别维度 (SH1=4, SH2=9 etc)
    cb_r = nn.Parameter(torch.from_numpy(data["cb_r"]).to(device).requires_grad_(True))
    cb_g = nn.Parameter(torch.from_numpy(data["cb_g"]).to(device).requires_grad_(True))
    cb_b = nn.Parameter(torch.from_numpy(data["cb_b"]).to(device).requires_grad_(True))

    # --- 加载 Indices (冻结，不可导) ---
    # 注意: Indices 必须是 LongTensor
    idx_r = torch.from_numpy(data["idx_r"].astype(np.int64)).to(device)
    idx_g = torch.from_numpy(data["idx_g"].astype(np.int64)).to(device)
    idx_b = torch.from_numpy(data["idx_b"].astype(np.int64)).to(device)

    # --- 加载其他属性 ---
    # Pos/Scale/Rot 保持冻结，防止破坏排序和空间结构
    xyz = torch.from_numpy(data["pos16"].astype(np.float32)).to(device)
    rot_q = torch.from_numpy(data["rot_q"]).to(device)  # 这里是量化后的整数，需解压

    # 解压 Rotation (简单的反量化用于渲染)
    rot_bits = int(data["rot_bits"])
    norm_factor = 32767.0 if rot_bits == 16 else 127.0
    rot = (rot_q.float() / norm_factor)
    rot = rot / (rot.norm(dim=1, keepdim=True) + 1e-9)
    rot = rot.requires_grad_(False)  # 冻结旋转

    scale_q = torch.from_numpy(data["sc_q"]).to(device)
    scale_mn = torch.tensor(data["sc_mn"]).to(device)
    scale_mx = torch.tensor(data["sc_mx"]).to(device)
    scale = scale_mn + (scale_mx - scale_mn) * (scale_q.float() / 255.0)
    scale = scale.requires_grad_(False)  # 冻结缩放

    # Opacity 建议微调，因为它对 SSIM 影响大且不影响压缩体积(只改值)
    op_q = torch.from_numpy(data["op_q"]).to(device)
    op_mn = torch.tensor(data["op_mn"]).to(device)
    op_mx = torch.tensor(data["op_mx"]).to(device)
    opacity = op_mn + (op_mx - op_mn) * (op_q.float() / 255.0)
    # 转换为 3DGS 内部的 logit 形式以便优化，或者直接优化值
    # 这里我们直接优化值，为了方便后续存回，限制在 [0,1]
    opacity = nn.Parameter(opacity.requires_grad_(True))

    print(f"[FT] Loaded {xyz.shape[0]} Gaussians.")
    print(f"[FT] Codebook shapes: R={cb_r.shape}, G={cb_g.shape}, B={cb_b.shape}")

    # 3. 初始化 Scene (加载相机)
    # 这是一个 Hack，我们需要一个 dummy GaussianModel 来骗过 Scene 加载器
    gaussians = GaussianModel(dataset.sh_degree)
    # 我们不调用 gaussians.load_ply，而是直接初始化 scene
    # 为了避免 Scene 报错找不到 ply，我们可能需要指向上一次训练的 ply 路径
    # 但 Scene 加载主要为了相机。
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=True)

    # 4. 优化器配置
    # 针对 Codebook 使用较大的学习率
    optimizer = optim.Adam([
        {'params': [cb_r, cb_g, cb_b], 'lr': 0.005, "name": "codebook"},
        {'params': [opacity], 'lr': 0.01, "name": "opacity"}
    ], lr=0.0)

    # 5. 训练循环
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    pbar = tqdm(range(args.finetune_iters), desc="Codebook Finetuning")

    for iteration in pbar:
        # 随机选相机
        viewpoint_cam = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras()) - 1)]

        # --- 动态构建特征 (On-the-fly Dequantization) ---
        # 查表得到 RGB 特征向量
        f_r = torch.index_select(cb_r, 0, idx_r)
        f_g = torch.index_select(cb_g, 0, idx_g)
        f_b = torch.index_select(cb_b, 0, idx_b)

        # 拼装回 GaussianModel 需要的格式
        # 假设 Encode 时结构是 [DC(1), Rest(N)]
        # R=[dc_r, rest_r...], G=[dc_g, rest_g...], B=[dc_b, rest_b...]

        # 提取 DC
        dc = torch.stack([f_r[:, 0], f_g[:, 0], f_b[:, 0]], dim=1).unsqueeze(1)  # [N, 1, 3]

        # 提取 Rest (如果有)
        if f_r.shape[1] > 1:
            # 拼装顺序需要匹配 gs_decode 的逻辑
            # gs_decode: rest = concat([r[:,1:], g[:,1:], b[:,1:]])
            # 我们这里还原成 [N, C, 1] 还是 [N, Total] ?
            # 3DGS render 里的 features_rest 通常是 [N, 15] (SH2) 或 [N, 9] (SH1)
            # 我们需要按照 "R的所有系数, G的所有系数, B的所有系数" 还是 "Coeff0_RGB, Coeff1_RGB..."?
            # 之前的 gs_encode/decode 逻辑是将 R/G/B 分开存。
            # 只要 gs_decode 读取顺序和这里拼装顺序一致，渲染就没问题。
            # 为了让标准 render 跑通，我们需要看 render 期待什么。
            # render 期待 features_rest 为 (N, (deg+1)^2-1, 3) ? 不，是 (N, 3 * n_coeffs) ?
            # 实际上 GaussianModel 存储的是 (N, n_features).
            # 让我们直接hack gaussians对象

            # 构造 Rest: [N, 3 * (D-1)]
            rest_r = f_r[:, 1:]
            rest_g = f_g[:, 1:]
            rest_b = f_b[:, 1:]
            # 简单拼接，只要维度对上，优化器会自动调整值去适应
            features_rest = torch.cat([rest_r, rest_g, rest_b], dim=1)
            # 注意：标准的 features_rest 形状通常需要 reshape，
            # 但既然我们是从零训练 codebook，只要映射关系固定，网络学会什么就是什么。
            # 唯一风险是如果 sh_degree > 0，renderer 会用球谐函数乘这些系数。
        else:
            features_rest = torch.zeros((xyz.shape[0], 0), device=device)

        # 赋值给 gaussians 对象 (Hack)
        gaussians._xyz = xyz
        gaussians._features_dc = dc
        gaussians._features_rest = features_rest.view(xyz.shape[0], -1)  # Flatten
        gaussians._opacity = opacity
        gaussians._scaling = scale
        gaussians._rotation = rot

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Backward
        loss.backward()

        # Step
        optimizer.step()
        optimizer.zero_grad()

        if iteration % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})

    # 6. 保存微调结果
    print(f"[FT] Saving finetuned model to {args.save_path}...")

    # 提取优化后的 numpy 值
    new_cb_r = cb_r.detach().cpu().numpy().astype(np.float16)
    new_cb_g = cb_g.detach().cpu().numpy().astype(np.float16)
    new_cb_b = cb_b.detach().cpu().numpy().astype(np.float16)

    # Opacity 需要重新量化回 8bit
    new_op = opacity.detach().clamp(0, 1)
    # 使用原来的 Min/Max 进行量化以保持兼容，或者重新计算 Min/Max
    # 为了简单且保持精度，建议更新 Min/Max
    new_op_val = new_op.cpu().numpy()
    op_mn_new = new_op_val.min()
    op_mx_new = new_op_val.max()
    # 避免除零
    span = max(op_mx_new - op_mn_new, 1e-9)
    new_op_q = np.round((new_op_val - op_mn_new) / span * 255.0).astype(np.uint8)

    # 另存为 NPZ
    np.savez_compressed(
        args.save_path,
        # 保持不变的
        degree=data["degree"],
        K=data["K"],
        morton_used=data["morton_used"],
        idx_r=data["idx_r"], idx_g=data["idx_g"], idx_b=data["idx_b"],
        pos16=data["pos16"],
        nrm16=data["nrm16"] if "nrm16" in data else None,
        sc_q=data["sc_q"], sc_mn=data["sc_mn"], sc_mx=data["sc_mx"], sc_bits=data["sc_bits"],
        rot_q=data["rot_q"], rot_bits=data["rot_bits"],
        # 更新的
        cb_r=new_cb_r, cb_g=new_cb_g, cb_b=new_cb_b,
        op_q=new_op_q, op_mn=op_mn_new.astype(np.float32), op_mx=op_mx_new.astype(np.float32), op_bits=data["op_bits"]
    )
    print("[FT] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--tag", required=True, help="Input NPZ tag (e.g. cb4096)")
    parser.add_argument("--save_path", default=None, help="Output NPZ path (default: overwrite)")
    parser.add_argument("--finetune_iters", type=int, default=2000)
    parser.add_argument("--iteration", type=int, default=30000)

    # 接收标准 3DGS 参数
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}",
                                      f"point_cloud.{args.tag}.finetuned.npz")

    finetune(lp.extract(args), op.extract(args), pp.extract(args), args)