# finetune.py
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import lpips


def training(dataset, opt, pipe, args):
    # 1. 初始化感知损失 (LPIPS)
    # 这里的 net='vgg' 是标准做法，能很好地衡量人眼感知的模糊度
    lpips_fn = lpips.LPIPS(net='vgg').cuda()

    # 2. 初始化高斯模型，注意这里的 sh_degree 要和你蒸馏后的模型一致 (SH2 则为 2)
    gaussians = GaussianModel(dataset.sh_degree)

    # 3. 初始化场景
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)

    # === 关键步骤：加载你已经压缩/蒸馏好的 PLY 文件 ===
    if args.start_checkpoint:
        print(f"Loading checkpoint from PLY: {args.start_checkpoint}")
        gaussians.load_ply(args.start_checkpoint)
    else:
        print("Error: Please provide path to your distilled PLY using --start_checkpoint")
        return

    # 4. 设置优化器参数 (关键：使用极低的学习率)
    # 将位置学习率设为原本的 1/10 甚至更低，防止破坏已经做好的几何结构
    opt.position_lr_init = 0.000016  # 原本是 0.00016
    opt.position_lr_final = 0.0000016
    opt.position_lr_delay_mult = 0.01
    opt.position_lr_max_steps = args.iterations

    # 颜色/SH 系数可以稍微给高一点，因为主要想修补纹理
    opt.feature_lr = 0.0025
    opt.opacity_lr = 0.05
    opt.scaling_lr = 0.005
    opt.rotation_lr = 0.001

    # 重新构建优化器
    gaussians.training_setup(opt)

    # 如果你想保持模型大小绝对不变，可以把下面这行设为 False (关闭分裂/克隆)
    # 但为了指标提升，建议开启，但阈值设高一点
    densification_enabled = False

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    print(f"Starting Fine-tuning for {args.iterations} iterations...")

    for iteration in range(1, args.iterations + 1):
        iter_start.record()

        # 挑选一个随机视角
        viewpoint_stack = scene.getTrainCameras()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # 获取 Ground Truth
        gt_image = viewpoint_cam.original_image.cuda()

        # === 核心修改：Loss 计算 ===
        # 原版 Loss
        Ll1 = l1_loss(image, gt_image)
        loss_ssim = 1.0 - ssim(image, gt_image)

        # 新增 LPIPS Loss (权重建议 0.1 ~ 0.5)
        # 注意：LPIPS 需要输入范围 [-1, 1]，而 3DGS 输出是 [0, 1]，需要转换
        loss_lpips = lpips_fn((image - 0.5) * 2, (gt_image - 0.5) * 2).mean()

        # 总 Loss：大幅增加 LPIPS 的权重来强行优化观感
        loss = (1.0 - opt.lambda_dssim) * Ll1 + \
               opt.lambda_dssim * loss_ssim + \
               0.2 * loss_lpips  # 0.2 是经验值，可调大

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 优化器步进
            if iteration < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # (可选) 如果你允许增加一点点点数来换取画质
            if densification_enabled and iteration < args.iterations - 500:
                # 只在误差极大的地方分裂，阈值设为原本(0.0002)的 2-3 倍
                gaussians.densify_and_prune(opt.densify_grad_threshold * 3.0, 0.005, scene.cameras_extent, 100)

            # 打印进度
            if iteration % 100 == 0:
                print(
                    f"Iter {iteration}: L1 {Ll1.item():.4f} | SSIM {1 - loss_ssim.item():.4f} | LPIPS {loss_lpips.item():.4f}")

    # 保存微调后的模型
    print("Saving fine-tuned model...")
    scene.save(iteration)
    print(f"Done. Model saved at output/{args.model_path}/point_cloud/iteration_{iteration}")


if __name__ == "__main__":
    # 参数解析
    parser = ArgumentParser(description="Fine-tune 3DGS post-distillation")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # === 修复点：删除下面这行，因为 OptimizationParams 已经定义了它 ===
    # parser.add_argument("--iterations", type=int, default=3000, help="Number of fine-tuning iterations")

    parser.add_argument("--start_checkpoint", type=str, required=True, help="Path to your distilled .ply file")

    args = parser.parse_args(sys.argv[1:])

    # 这里的 model_path 是为了输出目录用
    if not args.model_path:
        args.model_path = "finetuned_output"

    print("Optimizing " + args.model_path)

    # 启动训练
    training(lp.extract(args), op.extract(args), pp.extract(args), args)