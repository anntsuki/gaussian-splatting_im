# finetune.py
# 完整修复版：包含 cfg_args 保存、文件夹自动创建、相机队列修复、参数冲突修复

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

# 尝试导入 tensorboard，如果没有也不强求
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    print("Warning: Tensorboard not found, skipping logging.")


def training(dataset, opt, pipe, args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from):
    # 1. 初始化感知损失 (LPIPS)
    lpips_fn = lpips.LPIPS(net='vgg').cuda()

    # 2. 初始化高斯模型
    # 注意：这里会使用传入的 --sh_degree。如果你的模型只有 SH1，请务必在命令行指定 --sh_degree 1
    gaussians = GaussianModel(dataset.sh_degree)

    # 3. 初始化场景
    # load_iteration=0 强制它以“从头开始”的模式初始化，避免去空文件夹找存档导致报错
    # 这里的 args.model_path 已经被下面的主函数创建好了，所以 Scene 写入 input.ply 不会报错
    scene = Scene(dataset, gaussians, load_iteration=0, shuffle=False)

    # === 关键步骤：加载你已经压缩/蒸馏好的 PLY 文件 ===
    if args.start_checkpoint:
        print(f"Loading checkpoint from PLY: {args.start_checkpoint}")
        gaussians.load_ply(args.start_checkpoint)
    else:
        print("Error: Please provide path to your distilled PLY using --start_checkpoint")
        return

    # 4. 设置优化器参数 (极低学习率微调)
    # 将位置学习率设为原本的 1/10 甚至更低，防止破坏已经做好的几何结构
    opt.position_lr_init = 0.000016
    opt.position_lr_final = 0.0000016
    opt.position_lr_delay_mult = 0.01
    opt.position_lr_max_steps = opt.iterations

    opt.feature_lr = 0.0025
    opt.opacity_lr = 0.05
    opt.scaling_lr = 0.005
    opt.rotation_lr = 0.001

    gaussians.training_setup(opt)

    # 默认关闭分裂，只进行参数微调，保持模型体积不变
    densification_enabled = False

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    print(f"Starting Fine-tuning for {opt.iterations} iterations...")

    # 初始化 Tensorboard
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)

    # === 相机队列初始化 ===
    viewpoint_stack = None

    first_iter = 0

    # 开始训练循环
    for iteration in range(first_iter + 1, opt.iterations + 1):
        iter_start.record()

        # === 修复：相机队列自动补货逻辑 ===
        # 如果队列空了，重新从 scene 获取一遍
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # 随机取一张图片
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # 获取 Ground Truth
        gt_image = viewpoint_cam.original_image.cuda()

        # === Loss 计算 ===
        Ll1 = l1_loss(image, gt_image)
        loss_ssim = 1.0 - ssim(image, gt_image)

        # LPIPS Loss (核心提升点)
        # 将 [0,1] 转换为 [-1,1] 供 LPIPS 使用
        loss_lpips = lpips_fn((image - 0.5) * 2, (gt_image - 0.5) * 2).mean()

        # 总 Loss：适当增加 LPIPS 权重
        loss = (1.0 - opt.lambda_dssim) * Ll1 + \
               opt.lambda_dssim * loss_ssim + \
               0.2 * loss_lpips

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 记录进度
            if iteration % 100 == 0:
                print(
                    f"Iter {iteration}: L1 {Ll1.item():.4f} | SSIM {1 - loss_ssim.item():.4f} | LPIPS {loss_lpips.item():.4f}")
                if tb_writer:
                    tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
                    tb_writer.add_scalar('train_loss_patches/lpips_loss', loss_lpips.item(), iteration)

            # 优化器步进
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # (可选) 如果你真的想开启分裂来救画质，可以在这里把 densification_enabled 改为 True
            if densification_enabled and iteration < opt.iterations - 500:
                gaussians.densify_and_prune(opt.densify_grad_threshold * 3.0, 0.005, scene.cameras_extent, 100)

        # 保存模型
        if (iteration in saving_iterations):
            print(f"\n[ITER {iteration}] Saving Gaussian Model...")
            scene.save(iteration)

    # 训练结束保存最终模型
    print(f"Saving final fine-tuned model at iteration {iteration}...")
    scene.save(iteration)
    print(f"Done. Model saved at {args.model_path}/point_cloud/iteration_{iteration}")


if __name__ == "__main__":
    # 参数解析
    parser = ArgumentParser(description="Fine-tune 3DGS post-distillation")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # 注意：OptimizationParams 里已经有 --iterations，不要重复定义
    parser.add_argument("--start_checkpoint", type=str, required=True, help="Path to your distilled .ply file")

    # 其他常用参数，保持和 train.py 一致
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint_iter", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    # 这里的 model_path 是为了输出目录用
    if not args.model_path:
        args.model_path = "finetuned_output"

    print("Optimizing " + args.model_path)

    # === 【修复】关键：手动创建输出目录，并保存 cfg_args ===
    # 这就是你说的“cfg都没有”的修复点
    os.makedirs(args.model_path, exist_ok=True)

    # 将当前的运行参数保存到 cfg_args 文件中
    # 这样你以后就能看到你是用什么参数跑的这个微调了
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    # ===================================================

    # 启动训练
    # 注意把 args.iterations 传给 optimization params，覆盖默认的 30000
    op_args = op.extract(args)
    op_args.iterations = args.iterations

    training(lp.extract(args), op_args, pp.extract(args), args, args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint_iter, False)