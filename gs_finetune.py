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
    # --- 加载 Codebooks ---
    # 必须强转 .float() (即 float32)，否则渲染器会报 "expected Float but found Half"
    cb_r = nn.Parameter(torch.from_numpy(data["cb_r"]).to(device).float().requires_grad_(True))
    cb_g = nn.Parameter(torch.from_numpy(data["cb_g"]).to(device).float().requires_grad_(True))
    cb_b = nn.Parameter(torch.from_numpy(data["cb_b"]).to(device).float().requires_grad_(True))

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

    # ------------------ 替换开始 ------------------
    for iteration in pbar:
        try:
            viewpoint_cam = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras()) - 1)]
        except:
            viewpoint_cam = scene.getTrainCameras()[0]

        # --- On-the-fly Decoding ---
        f_r = torch.index_select(cb_r, 0, idx_r)
        f_g = torch.index_select(cb_g, 0, idx_g)
        f_b = torch.index_select(cb_b, 0, idx_b)

        # 组装 DC: [N, 1, 3]  <-- 这是一个 3维张量
        dc = torch.stack([f_r[:, 0], f_g[:, 0], f_b[:, 0]], dim=1).unsqueeze(1)

        # 组装 Rest: 必须也是 3维张量 [N, Coeffs, 3]
        if f_r.shape[1] > 1:
            rest_r = f_r[:, 1:]  # [N, C]
            rest_g = f_g[:, 1:]  # [N, C]
            rest_b = f_b[:, 1:]  # [N, C]

            # === 关键修改点 ===
            # 不要用 cat(dim=1) 变成 [N, 3*C]，而是用 stack(dim=-1) 变成 [N, C, 3]
            features_rest = torch.stack([rest_r, rest_g, rest_b], dim=-1)
        else:
            # 如果没有 rest，也要保证是 3维的空张量 [N, 0, 3]
            features_rest = torch.zeros((xyz.shape[0], 0, 3), device=device)

        # 赋值给 GaussianModel (不要 view Flatten!)
        gaussians._xyz = xyz
        gaussians._features_dc = dc
        gaussians._features_rest = features_rest  # 直接赋值 3D 张量
        gaussians._opacity = opacity
        gaussians._scaling = scale
        gaussians._rotation = rot

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        # Loss (L1 + SSIM)
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iteration % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})
    # ------------------ 替换结束 ------------------

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

    # --- 修复核心：删除冲突的参数定义 ---
    # ModelParams 会自动添加 --model_path 和 --source_path，所以这里不能再add了
    # parser.add_argument("--model_path", required=True)  <-- 删除
    # parser.add_argument("--source_path", required=True) <-- 删除

    # 只保留脚本特有的参数
    parser.add_argument("--tag", required=True, help="Input NPZ tag (e.g. cb4096)")
    parser.add_argument("--save_path", default=None, help="Output NPZ path (default: overwrite)")
    parser.add_argument("--finetune_iters", type=int, default=2000)
    parser.add_argument("--iteration", type=int, default=30000)

    # 加载标准 3DGS 参数 (这里会自动把 -s, -m 加进去)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args()

    # 检查必要的路径参数 (因为 ModelParams 里可能有默认值，这里手动检查一下更稳)
    if not args.model_path:
        parser.error("argument --model_path/-m is required")
    if not args.source_path:
        parser.error("argument --source_path/-s is required")

    # 自动生成 save_path
    if args.save_path is None:
        args.save_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}",
                                      f"point_cloud.{args.tag}.finetuned.npz")

    finetune(lp.extract(args), op.extract(args), pp.extract(args), args)