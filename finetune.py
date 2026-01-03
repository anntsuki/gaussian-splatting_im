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
    lpips_fn = lpips.LPIPS(net='vgg').cuda()

    # 2. 初始化高斯模型 (会使用传入的 --sh_degree)
    gaussians = GaussianModel(dataset.sh_degree)

    # === 修复点A：手动创建输出目录，防止报错 ===
    import os
    os.makedirs(args.model_path, exist_ok=True)

    # 3. 初始化场景
    # === 修复点B：load_iteration=0 强制从头初始化，避免去空文件夹找存档报错 ===
    scene = Scene(dataset, gaussians, load_iteration=0, shuffle=False)

    # === 关键步骤：加载你已经压缩/蒸馏好的 PLY 文件 ===
    if args.start_checkpoint:
        print(f"Loading checkpoint from PLY: {args.start_checkpoint}")
        gaussians.load_ply(args.start_checkpoint)
    else:
        print("Error: Please provide path to your distilled PLY using --start_checkpoint")
        return

    # 4. 设置优化器参数 (极低学习率微调)
    opt.position_lr_init = 0.000016
    opt.position_lr_final = 0.0000016
    opt.position_lr_delay_mult = 0.01
    opt.position_lr_max_steps = args.iterations

    opt.feature_lr = 0.0025
    opt.opacity_lr = 0.05
    opt.scaling_lr = 0.005
    opt.rotation_lr = 0.001

    gaussians.training_setup(opt)

    # 默认关闭分裂，只进行参数微调
    densification_enabled = False

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    print(f"Starting Fine-tuning for {args.iterations} iterations...")

    # === 修复点C：初始化相机堆栈为 None ===
    viewpoint_stack = None

    for iteration in range(1, args.iterations + 1):
        iter_start.record()

        # === 修复点D：相机队列自动补货逻辑 ===
        # 如果手里没照片了，就从 Scene 里重新拷贝一份完整的训练集
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # 随机取一张
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # 获取 Ground Truth
        gt_image = viewpoint_cam.original_image.cuda()

        # Loss 计算
        Ll1 = l1_loss(image, gt_image)
        loss_ssim = 1.0 - ssim(image, gt_image)

        # LPIPS Loss
        loss_lpips = lpips_fn((image - 0.5) * 2, (gt_image - 0.5) * 2).mean()

        # 总 Loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + \
               opt.lambda_dssim * loss_ssim + \
               0.2 * loss_lpips

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if iteration < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # (可选) 如果需要开启分裂，在此处控制
            if densification_enabled and iteration < args.iterations - 500:
                gaussians.densify_and_prune(opt.densify_grad_threshold * 3.0, 0.005, scene.cameras_extent, 100)

            # 打印进度
            if iteration % 100 == 0:
                print(
                    f"Iter {iteration}: L1 {Ll1.item():.4f} | SSIM {1 - loss_ssim.item():.4f} | LPIPS {loss_lpips.item():.4f}")

    # 保存微调后的模型
    print("Saving fine-tuned model...")
    scene.save(iteration)
    print(f"Done. Model saved at {args.model_path}/point_cloud/iteration_{iteration}")


if __name__ == "__main__":
    # 参数解析
    parser = ArgumentParser(description="Fine-tune 3DGS post-distillation")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # === 修复点E：删除了重复定义的 --iterations，直接使用 OptimizationParams 里的 ===

    parser.add_argument("--start_checkpoint", type=str, required=True, help="Path to your distilled .ply file")

    args = parser.parse_args(sys.argv[1:])

    if not args.model_path:
        args.model_path = "finetuned_output"

    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args), args)