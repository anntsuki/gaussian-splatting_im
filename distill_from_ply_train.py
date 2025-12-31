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


def sh_rest_dim(deg: int) -> int:
    return (deg + 1) ** 2 - 1  # exclude DC


@torch.no_grad()
def force_truncate_sh(student: GaussianModel, target_sh: int):
    """
    强制把 features_rest 截断到 target_sh 对应维度。
    兼容 [N,3,K] 和 [N,K,3] 两种布局（RGB=3 的那一维不动，截断另一维）。
    """
    k = sh_rest_dim(target_sh)
    if not hasattr(student, "_features_rest"):
        raise RuntimeError("GaussianModel has no _features_rest")

    fr = student._features_rest
    if fr.ndim != 3:
        raise RuntimeError(f"_features_rest expect 3D, got {tuple(fr.shape)}")

    # 判断哪一维是 RGB(=3)
    if fr.shape[1] == 3 and fr.shape[2] != 3:
        coeff_axis = 2  # [N,3,K]
    elif fr.shape[2] == 3 and fr.shape[1] != 3:
        coeff_axis = 1  # [N,K,3]
    else:
        coeff_axis = 1 if fr.shape[1] > fr.shape[2] else 2  # fallback

    coeff_dim = fr.shape[coeff_axis]
    if coeff_dim < k:
        raise RuntimeError(f"SH coeff dim {coeff_dim} < needed {k}, shape={tuple(fr.shape)}, coeff_axis={coeff_axis}")

    new_fr = fr[:, :k, :].contiguous() if coeff_axis == 1 else fr[:, :, :k].contiguous()
    student._features_rest = torch.nn.Parameter(new_fr)


def patch_cfg_args_sh_degree(cfg_path: str, new_sh: int):
    """把输出目录 cfg_args 里的 sh_degree/max_sh_degree 改成 new_sh，避免 benchmark 读错"""
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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_teacher_student_from_ply(dataset, pipe, teacher_model_path: str, iteration: int, target_sh: int):
    """
    Teacher/Student 都从同一个 PLY 初始化：
      teacher = 原 SH（cfg_args 里的 sh_degree）
      student = 从 teacher copy 的 PLY，降阶到 target_sh
    返回: teacher_scene, teacher, student
    """
    teacher_ds = copy.deepcopy(dataset)
    teacher_ds.model_path = teacher_model_path

    teacher = GaussianModel(teacher_ds.sh_degree)
    teacher_scene = Scene(teacher_ds, teacher, load_iteration=iteration, shuffle=False)

    student = GaussianModel(teacher_ds.sh_degree)
    _ = Scene(teacher_ds, student, load_iteration=iteration, shuffle=False)

    # 降阶：优先用 onedownSHdegree（如果有），再强制截断兜底
    with torch.no_grad():
        if hasattr(student, "onedownSHdegree"):
            try:
                student.max_sh_degree = target_sh
            except Exception:
                pass
            for _ in range(3):
                try:
                    student.onedownSHdegree()
                except Exception:
                    break
        force_truncate_sh(student, target_sh)

    return teacher_scene, teacher, student


def distill_train_sh_from_ply(
    dataset,
    pipe,
    teacher_model_path: str,
    out_model_path: str,
    iteration: int = 30000,
    save_iteration: int = 30000,
    target_sh: int = 2,
    iters: int = 12000,
    lr_sh: float = 3e-3,
    lambda_dssim: float = 0.2,
    use_augmented_view: bool = True,
    pseudo_start_ratio: float = 0.7,
    pv_every: int = 4,
    pv_trans: float = 0.03,
    pv_rot: float = 0.0,
    w_gt: float = 0.7,
    enable_covariance: bool = False,
    enable_opacity: bool = False,
):
    """
    路线A（最接近 LightGaussian 的 SH distillation）：
    - teacher 从 PLY（SH=3）加载
    - student 从同一 PLY 初始化，降到 SH=2
    - loss = teacher 蒸馏 + (真实视角) GT 回补
    - pseudo-view 在后半段开启，且频率可控（避免一开始学歪）
    """
    device = "cuda"
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    teacher_scene, teacher, student = build_teacher_student_from_ply(dataset, pipe, teacher_model_path, iteration, target_sh)

    # 冻结非 SH
    if hasattr(student, "_xyz"): student._xyz.requires_grad_(False)
    if hasattr(student, "_scaling"): student._scaling.requires_grad_(enable_covariance)
    if hasattr(student, "_rotation"): student._rotation.requires_grad_(enable_covariance)
    if hasattr(student, "_opacity"): student._opacity.requires_grad_(enable_opacity)
    student._features_dc.requires_grad_(True)
    student._features_rest.requires_grad_(True)

    # 只训练 SH
    optim = torch.optim.Adam(
        [{"params": [student._features_dc], "lr": lr_sh},
         {"params": [student._features_rest], "lr": lr_sh}]
    )

    cams = teacher_scene.getTrainCameras()
    assert len(cams) > 0, "No train cameras"
    use_exp = getattr(dataset, "train_test_exp", False)

    print(f"[INFO] Distill PLY: {teacher_model_path} iter={iteration} | SH->{target_sh} | steps={iters}")
    print(f"[INFO] GT weight={w_gt}, pseudo={use_augmented_view}, pseudo_start_ratio={pseudo_start_ratio}, pv_every={pv_every}, pv_trans={pv_trans}")

    pbar = tqdm(range(1, iters + 1), desc="Distill SH")
    for step in pbar:
        cam_org = cams[randint(0, len(cams) - 1)]
        cam = copy.deepcopy(cam_org)

        # pseudo view：后半段才开，且频率 pv_every 控制
        use_pseudo = False
        if use_augmented_view and step > int(iters * pseudo_start_ratio):
            use_pseudo = (step % pv_every != 0)
        if use_pseudo:
            cam = gaussian_poses(cam, mean=0, std_dev_translation=pv_trans, std_dev_rotation=pv_rot)

        with torch.no_grad():
            t_img = render(cam, teacher, pipe, background, use_trained_exp=use_exp)["render"].detach()

        s_img = render(cam, student, pipe, background, use_trained_exp=use_exp)["render"]

        # teacher distill
        l1_t = l1_loss(s_img, t_img)
        loss = (1.0 - lambda_dssim) * l1_t + lambda_dssim * (1.0 - ssim(s_img, t_img))

        # GT 回补（只在真实视角）
        if (not use_pseudo) and hasattr(cam_org, "original_image") and (cam_org.original_image is not None):
            gt = cam_org.original_image.to(device)
            l1_gt = l1_loss(s_img, gt)
            loss_gt = (1.0 - lambda_dssim) * l1_gt + lambda_dssim * (1.0 - ssim(s_img, gt))
            loss = loss + w_gt * loss_gt

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 50 == 0:
            pbar.set_postfix(loss=float(loss.item()), l1_teacher=float(l1_t.item()), pseudo=int(use_pseudo))

    # ---- save output model (cfg_args / exposure.json / cameras.json + point_cloud/iteration_x/point_cloud.ply)
    ensure_dir(out_model_path)

    # copy meta files
    for fname in ["cfg_args", "cameras.json", "exposure.json"]:
        src = os.path.join(teacher_model_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_model_path, fname))

    # patch cfg_args SH
    cfg_dst = os.path.join(out_model_path, "cfg_args")
    patch_cfg_args_sh_degree(cfg_dst, target_sh)

    pc_dir = os.path.join(out_model_path, "point_cloud", f"iteration_{save_iteration}")
    ensure_dir(pc_dir)
    out_ply = os.path.join(pc_dir, "point_cloud.ply")
    student.save_ply(out_ply)

    print("[DONE] Saved:")
    print("  model:", out_model_path)
    print("  ply  :", out_ply)


def main():
    parser = ArgumentParser("SH distillation from PLY (no .pth), with GT + pseudo-view")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--teacher_model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--save_iteration", type=int, default=30000)

    parser.add_argument("--new_max_sh", type=int, default=2)
    parser.add_argument("--iters", type=int, default=12000)
    parser.add_argument("--lr_sh", type=float, default=3e-3)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)

    parser.add_argument("--w_gt", type=float, default=0.7)
    parser.add_argument("--augmented_view", action="store_true")
    parser.add_argument("--pseudo_start_ratio", type=float, default=0.7)
    parser.add_argument("--pv_every", type=int, default=4)
    parser.add_argument("--pv_trans", type=float, default=0.03)
    parser.add_argument("--pv_rot", type=float, default=0.0)

    parser.add_argument("--enable_covariance", action="store_true")
    parser.add_argument("--enable_opacity", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    torch.cuda.set_device(0)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    distill_train_sh_from_ply(
        dataset=dataset,
        pipe=pipe,
        teacher_model_path=args.teacher_model_path,
        out_model_path=args.model_path,
        iteration=args.iteration,
        save_iteration=args.save_iteration,
        target_sh=args.new_max_sh,
        iters=args.iters,
        lr_sh=args.lr_sh,
        lambda_dssim=args.lambda_dssim,
        use_augmented_view=args.augmented_view,
        pseudo_start_ratio=args.pseudo_start_ratio,
        pv_every=args.pv_every,
        pv_trans=args.pv_trans,
        pv_rot=args.pv_rot,
        w_gt=args.w_gt,
        enable_covariance=args.enable_covariance,
        enable_opacity=args.enable_opacity,
    )


if __name__ == "__main__":
    main()
