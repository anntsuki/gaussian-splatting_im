import os, sys, copy, re, shutil
import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import gaussian_poses
from utils.image_utils import psnr


def sh_rest_dim(deg: int) -> int:
    return (deg + 1) ** 2 - 1  # exclude DC


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def patch_cfg_args_sh_degree(cfg_path: str, new_sh: int):
    if not os.path.exists(cfg_path):
        return
    s = open(cfg_path, "r", encoding="utf-8", errors="ignore").read()
    s2 = s
    s2 = re.sub(r"(sh_degree\s*=\s*)\d+", rf"\g<1>{new_sh}", s2)
    s2 = re.sub(r"(sh_degree\s*:\s*)\d+", rf"\g<1>{new_sh}", s2)
    s2 = re.sub(r"(max_sh_degree\s*=\s*)\d+", rf"\g<1>{new_sh}", s2)
    s2 = re.sub(r"(max_sh_degree\s*:\s*)\d+", rf"\g<1>{new_sh}", s2)
    if s2 != s:
        open(cfg_path, "w", encoding="utf-8").write(s2)


def get_gt_image(cam):
    """兼容不同分支 Camera 的 GT 字段命名"""
    for name in ["original_image", "image", "gt_image", "gt"]:
        if hasattr(cam, name):
            im = getattr(cam, name)
            if im is not None:
                return im
    return None


@torch.no_grad()
def force_truncate_sh(student: GaussianModel, target_sh: int):
    """
    【修复核心】强制截断 SH 系数，保留低频部分，并更新模型属性。
    """
    target_k = sh_rest_dim(target_sh)
    fr = student._features_rest

    # 检测维度: [N, K, 3] or [N, 3, K]
    if fr.ndim != 3:
        raise RuntimeError(f"features_rest should be 3D, got {tuple(fr.shape)}")

    # 判定哪个轴是 SH 系数轴
    if fr.shape[1] > fr.shape[2]:
        coeff_axis = 1
    else:
        coeff_axis = 2

    current_k = fr.shape[coeff_axis]

    if current_k <= target_k:
        print(f"[WARNING] Student SH dim ({current_k}) is already <= target ({target_k}). Skip truncation.")
        # 即使维度没变，也要确保 active_degree 是对的
        student.max_sh_degree = target_sh
        student.active_sh_degree = target_sh
        return

    print(f"[INFO] Truncating SH from {current_k} to {target_k} channels (Keeping Head)...")

    # 只取前 target_k 个系数 (低频信息)
    if coeff_axis == 1:
        new_fr = fr[:, :target_k, :].contiguous()
    else:
        new_fr = fr[:, :, :target_k].contiguous()

    # 更新参数
    student._features_rest = torch.nn.Parameter(new_fr)

    # 【关键】必须更新这两个属性，否则光栅化器会读错显存导致花屏
    student.max_sh_degree = target_sh
    student.active_sh_degree = target_sh


def main():
    parser = ArgumentParser("SH distillation from PLY")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--teacher_model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--save_iteration", type=int, default=30000)

    parser.add_argument("--new_max_sh", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--iters", type=int, default=2000)  # 建议2000次够了
    parser.add_argument("--lr_sh", type=float, default=0.0025)  # 标准 3DGS 学习率
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--w_gt", type=float, default=0.7)

    parser.add_argument("--augmented_view", action="store_true")
    parser.add_argument("--pseudo_start_ratio", type=float, default=0.7)
    parser.add_argument("--pv_every", type=int, default=4)
    parser.add_argument("--pv_trans", type=float, default=0.03)
    parser.add_argument("--pv_rot", type=float, default=0.0)

    args = parser.parse_args(sys.argv[1:])
    torch.cuda.set_device(0)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    device = "cuda"
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # ---- load teacher from PLY
    teacher_ds = copy.deepcopy(dataset)
    teacher_ds.model_path = args.teacher_model_path
    teacher = GaussianModel(teacher_ds.sh_degree)
    teacher_scene = Scene(teacher_ds, teacher, load_iteration=args.iteration, shuffle=False)

    # ---- load student from same PLY
    student = GaussianModel(teacher_ds.sh_degree)
    _ = Scene(teacher_ds, student, load_iteration=args.iteration, shuffle=False)

    use_exp = getattr(dataset, "train_test_exp", False)

    cams = teacher_scene.getTrainCameras()
    assert len(cams) > 0, "No train cameras"

    # 【修复】强制截断并更新 Degree
    with torch.no_grad():
        force_truncate_sh(student, args.new_max_sh)

    # freeze geometry, train SH only
    if hasattr(student, "_xyz"): student._xyz.requires_grad_(False)
    if hasattr(student, "_scaling"): student._scaling.requires_grad_(False)
    if hasattr(student, "_rotation"): student._rotation.requires_grad_(False)
    if hasattr(student, "_opacity"): student._opacity.requires_grad_(False)

    # 只训练颜色
    student._features_dc.requires_grad_(True)
    student._features_rest.requires_grad_(True)

    # 这里的 lr 用标准的 0.0025 会稳一点
    optim = torch.optim.Adam(
        [{"params": [student._features_dc], "lr": args.lr_sh},
         {"params": [student._features_rest], "lr": args.lr_sh}]
    )

    print(f"[INFO] Start distill: target_sh={args.new_max_sh}, iters={args.iters}, lr={args.lr_sh}")

    pbar = tqdm(range(1, args.iters + 1), desc="Distill SH")
    for step in pbar:
        # 随机选相机
        cam_idx = randint(0, len(cams) - 1)
        cam_org = cams[cam_idx]
        cam = copy.deepcopy(cam_org)

        # pseudo data augmentation
        use_pseudo = False
        if args.augmented_view and step > int(args.iters * args.pseudo_start_ratio):
            use_pseudo = (step % args.pv_every != 0)
        if use_pseudo:
            cam = gaussian_poses(cam, mean=0, std_dev_translation=args.pv_trans, std_dev_rotation=args.pv_rot)

        # Forward Teacher
        with torch.no_grad():
            t_img = render(cam, teacher, pipe, background, use_trained_exp=use_exp)["render"].detach()

        # Forward Student
        s_img = render(cam, student, pipe, background, use_trained_exp=use_exp)["render"]

        # Loss 1: Teacher Distillation
        l1_t = l1_loss(s_img, t_img)
        loss = (1.0 - args.lambda_dssim) * l1_t + args.lambda_dssim * (1.0 - ssim(s_img, t_img))

        # Loss 2: GT Supervision (Only on real views)
        if not use_pseudo:
            gt = get_gt_image(cam_org)
            if gt is not None:
                gt = gt.to(device)
                l1_gt = l1_loss(s_img, gt)
                loss_gt = (1.0 - args.lambda_dssim) * l1_gt + args.lambda_dssim * (1.0 - ssim(s_img, gt))
                loss = loss + args.w_gt * loss_gt

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 50 == 0:
            pbar.set_postfix(loss=float(loss.item()), l1_teacher=float(l1_t.item()))

    # ---- save output
    out_dir = args.model_path  # 这里的 model_path 其实是 output path
    ensure_dir(out_dir)

    # copy meta files
    for fname in ["cfg_args", "cameras.json", "exposure.json"]:
        src = os.path.join(args.teacher_model_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))

    # patch cfg_args sh_degree
    cfg_dst = os.path.join(out_dir, "cfg_args")
    patch_cfg_args_sh_degree(cfg_dst, args.new_max_sh)

    pc_dir = os.path.join(out_dir, "point_cloud", f"iteration_{args.save_iteration}")
    ensure_dir(pc_dir)
    out_ply = os.path.join(pc_dir, "point_cloud.ply")
    student.save_ply(out_ply)

    print("[DONE] Saved distilled model:")
    print("  model:", out_dir)
    print("  ply  :", out_ply)


if __name__ == "__main__":
    main()