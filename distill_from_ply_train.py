# distill_from_ply_train.py
import os
import sys
import copy
import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render, network_gui
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import gaussian_poses
from utils.logger_utils import prepare_output_and_logger, training_report

def training_from_ply(args, dataset, opt, pipe, testing_iterations, saving_iterations,
                      debug_from, new_max_sh, teacher_model_path):
    """
    Teacher & Student 都从 PLY 加载，不依赖任何 .pth checkpoint。
    Teacher: SH=old_sh
    Student: 从 teacher 参数初始化，然后降阶到 new_max_sh（如 2），只蒸馏训练 SH（默认不冻结 opacity/cov，可用开关冻结）
    """
    first_iter = 0

    # 记录原 SH，Student 目标 SH
    old_sh_degree = dataset.sh_degree
    dataset.sh_degree = new_max_sh

    # 输出目录 & cfg_args
    tb_writer = prepare_output_and_logger(dataset)

    # ---- Load teacher from PLY (model_path = teacher_model_path)
    with torch.no_grad():
        teacher_gaussians = GaussianModel(old_sh_degree)
    teacher_dataset = copy.deepcopy(dataset)
    teacher_dataset.model_path = teacher_model_path
    teacher_scene = Scene(teacher_dataset, teacher_gaussians, load_iteration=args.iteration, shuffle=False)
    teacher_gaussians.optimizer = None  # teacher 不训练

    # ---- Build student: init from teacher params (capture/restore)
    student_gaussians = GaussianModel(old_sh_degree)
    student_scene = Scene(dataset, student_gaussians)  # output model_path = args.model_path

    # 用 teacher 的参数初始化 student
    with torch.no_grad():
        student_gaussians.restore(teacher_gaussians.capture(), copy.deepcopy(opt))

    # 降 SH 阶：3->2（按你仓库里的实现，通常 oneDown 一次即可）
    student_gaussians.max_sh_degree = new_max_sh
    if hasattr(student_gaussians, "onedownSHdegree"):
        # 多降几次也安全：直到 max_sh_degree == new_max_sh
        # 注意：有些实现 onedownSHdegree 只做一次降阶，不更新 max_sh_degree
        # 所以我们最多循环 3 次兜底
        for _ in range(3):
            try:
                student_gaussians.onedownSHdegree()
            except Exception:
                break
    else:
        raise RuntimeError("你的 GaussianModel 里没有 onedownSHdegree()，无法自动降阶。")

    # 设置训练
    student_gaussians.training_setup(opt)

    # 可选冻结：协方差/不透明度（与你 distill_train.py 同款开关）
    if (not args.enable_covariance):
        if hasattr(student_gaussians, "_scaling"):
            student_gaussians._scaling.requires_grad = False
        if hasattr(student_gaussians, "_rotation"):
            student_gaussians._rotation.requires_grad = False

    if (not args.enable_opacity):
        if hasattr(student_gaussians, "_opacity"):
            student_gaussians._opacity.requires_grad = False

    # background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc=f"Distill (PLY) progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        # GUI 相关：你不用 GUI 也没事，保持兼容
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, student_gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        student_gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = student_scene.getTrainCameras().copy()

        viewpoint_cam_org = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam = copy.deepcopy(viewpoint_cam_org)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # pseudo view（与你贴的 distill_train 同款逻辑）
        if args.augmented_view and (iteration % 3):
            viewpoint_cam = gaussian_poses(
                viewpoint_cam, mean=0, std_dev_translation=args.pv_trans, std_dev_rotation=args.pv_rot
            )

        # student render
        student_image = render(viewpoint_cam, student_gaussians, pipe, background)["render"]
        # teacher render (detach)
        with torch.no_grad():
            teacher_image = render(viewpoint_cam, teacher_gaussians, pipe, background)["render"].detach()

        Ll1 = l1_loss(student_image, teacher_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(student_image, teacher_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 保存（默认保存到 iteration=args.iteration，方便你 benchmark 继续用 30000）
            if (iteration in saving_iterations) or (iteration == opt.iterations):
                save_it = args.save_iteration
                print(f"\n[ITER {iteration}] Saving distilled Gaussians as iteration_{save_it}")
                student_scene.save(save_it)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations,
                            student_scene, render, (pipe, background))

            if iteration < opt.iterations:
                student_gaussians.optimizer.step()
                student_gaussians.optimizer.zero_grad(set_to_none=True)

def main():
    parser = ArgumentParser(description="PLY-based SH distillation (no .pth needed)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # extra args
    parser.add_argument("--teacher_model_path", type=str, required=True,
                        help="Teacher 模型目录（里面有 point_cloud/iteration_x/point_cloud.ply）")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="从 teacher 的 iteration_XXX 读取 PLY")
    parser.add_argument("--save_iteration", type=int, default=30000,
                        help="把蒸馏后的 ply 保存为 iteration_XXX（默认 30000，方便直接用你原评测命令）")
    parser.add_argument("--new_max_sh", type=int, default=2)
    parser.add_argument("--augmented_view", action="store_true")
    parser.add_argument("--pv_trans", type=float, default=0.05)
    parser.add_argument("--pv_rot", type=float, default=0.0)

    # keep same flags as your distill_train
    parser.add_argument("--enable_covariance", action="store_true")
    parser.add_argument("--enable_opacity", action="store_true")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    # 你不需要 test/save/checkpoint 的复杂逻辑，保留接口兼容
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1])
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    print("Distill output model_path:", args.model_path)
    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # 训练时用 opt.iterations 控制“蒸馏步数”
    training_from_ply(
        args, dataset, opt, pipe,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        debug_from=args.debug_from,
        new_max_sh=args.new_max_sh,
        teacher_model_path=args.teacher_model_path
    )

    print("\nDistill complete.")

if __name__ == "__main__":
    main()
