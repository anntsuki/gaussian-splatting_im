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
def detect_coeff_axis(fr: torch.Tensor) -> int:
    """
    fr is [N, ?, ?], one axis is RGB=3, the other is coeff K
    return coeff_axis (1 or 2)
    """
    if fr.ndim != 3:
        raise RuntimeError(f"features_rest should be 3D, got {tuple(fr.shape)}")
    if fr.shape[1] == 3 and fr.shape[2] != 3:
        return 2  # [N,3,K]
    if fr.shape[2] == 3 and fr.shape[1] != 3:
        return 1  # [N,K,3]
    # fallback: choose larger as coeff axis
    return 1 if fr.shape[1] > fr.shape[2] else 2


@torch.no_grad()
def truncate_features_rest(fr: torch.Tensor, k: int, coeff_axis: int, mode: str) -> torch.Tensor:
    """
    mode: "head" keep first k coeffs; "tail" keep last k coeffs
    """
    if coeff_axis == 1:
        if mode == "head":
            return fr[:, :k, :].contiguous()
        else:
            return fr[:, -k:, :].contiguous()
    else:
        if mode == "head":
            return fr[:, :, :k].contiguous()
        else:
            return fr[:, :, -k:].contiguous()


@torch.no_grad()
def auto_choose_truncation(student: GaussianModel, teacher: GaussianModel, cams, pipe, background, use_exp: bool, target_sh: int, n_probe: int = 5):
    """
    在 head vs tail 两种截断方式中，选一个让 student 初始渲染更像 teacher 的。
    返回 chosen_mode in {"head","tail"}，并把 student._features_rest 设置为 chosen 截断结果。
    """
    k = sh_rest_dim(target_sh)
    fr = student._features_rest
    coeff_axis = detect_coeff_axis(fr)

    if (fr.shape[coeff_axis] < k):
        raise RuntimeError(f"coeff dim too small: shape={tuple(fr.shape)}, coeff_axis={coeff_axis}, need={k}")

    # prepare probes
    picks = []
    for _ in range(min(n_probe, len(cams))):
        picks.append(cams[randint(0, len(cams) - 1)])

    # evaluate head
    fr_head = truncate_features_rest(fr, k, coeff_axis, "head")
    student._features_rest = torch.nn.Parameter(fr_head)
    psnrs_head = []
    for cam in picks:
        t = render(cam, teacher, pipe, background, use_trained_exp=use_exp)["render"].detach()
        s = render(cam, student, pipe, background, use_trained_exp=use_exp)["render"].detach()
        psnrs_head.append(float(psnr(s, t)))
    mean_head = sum(psnrs_head) / len(psnrs_head)

    # evaluate tail
    fr_tail = truncate_features_rest(fr, k, coeff_axis, "tail")
    student._features_rest = torch.nn.Parameter(fr_tail)
    psnrs_tail = []
    for cam in picks:
        t = render(cam, teacher, pipe, background, use_trained_exp=use_exp)["render"].detach()
        s = render(cam, student, pipe, background, use_trained_exp=use_exp)["render"].detach()
        psnrs_tail.append(float(psnr(s, t)))
    mean_tail = sum(psnrs_tail) / len(psnrs_tail)

    if mean_head >= mean_tail:
        student._features_rest = torch.nn.Parameter(fr_head)
        chosen = "head"
        best = mean_head
    else:
        student._features_rest = torch.nn.Parameter(fr_tail)
        chosen = "tail"
        best = mean_tail

    print(f"[AUTO] truncation choose={chosen} (PSNR vs teacher: head={mean_head:.3f}, tail={mean_tail:.3f}, best={best:.3f})")
    return chosen


def main():
    parser = ArgumentParser("SH distillation from PLY (no .pth), robust truncation + GT")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--teacher_model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--save_iteration", type=int, default=30000)

    parser.add_argument("--new_max_sh", type=int, default=2, choices=[1,2,3])
    parser.add_argument("--iters", type=int, default=12000)
    parser.add_argument("--lr_sh", type=float, default=3e-3)
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

    # ---- load student from same PLY, then down-sh
    student = GaussianModel(teacher_ds.sh_degree)
    _ = Scene(teacher_ds, student, load_iteration=args.iteration, shuffle=False)

    use_exp = getattr(dataset, "train_test_exp", False)

    # prefer onedownSHdegree if exists
    with torch.no_grad():
        if hasattr(student, "onedownSHdegree"):
            try:
                student.max_sh_degree = args.new_max_sh
            except Exception:
                pass
            for _ in range(3):
                try:
                    student.onedownSHdegree()
                except Exception:
                    break

    cams = teacher_scene.getTrainCameras()
    assert len(cams) > 0, "No train cameras"

    # critical: choose correct truncation order (head vs tail)
    with torch.no_grad():
        auto_choose_truncation(student, teacher, cams, pipe, background, use_exp, args.new_max_sh, n_probe=5)

    # freeze geometry, train SH only
    if hasattr(student, "_xyz"): student._xyz.requires_grad_(False)
    if hasattr(student, "_scaling"): student._scaling.requires_grad_(False)
    if hasattr(student, "_rotation"): student._rotation.requires_grad_(False)
    if hasattr(student, "_opacity"): student._opacity.requires_grad_(False)
    student._features_dc.requires_grad_(True)
    student._features_rest.requires_grad_(True)

    optim = torch.optim.Adam(
        [{"params": [student._features_dc], "lr": args.lr_sh},
         {"params": [student._features_rest], "lr": args.lr_sh}]
    )

    print(f"[INFO] Start distill: iters={args.iters}, lr_sh={args.lr_sh}, w_gt={args.w_gt}, pseudo={args.augmented_view}")

    pbar = tqdm(range(1, args.iters + 1), desc="Distill SH")
    for step in pbar:
        cam_org = cams[randint(0, len(cams) - 1)]
        cam = copy.deepcopy(cam_org)

        # pseudo late + controlled frequency
        use_pseudo = False
        if args.augmented_view and step > int(args.iters * args.pseudo_start_ratio):
            use_pseudo = (step % args.pv_every != 0)
        if use_pseudo:
            cam = gaussian_poses(cam, mean=0, std_dev_translation=args.pv_trans, std_dev_rotation=args.pv_rot)

        with torch.no_grad():
            t_img = render(cam, teacher, pipe, background, use_trained_exp=use_exp)["render"].detach()

        s_img = render(cam, student, pipe, background, use_trained_exp=use_exp)["render"]

        # teacher distill
        l1_t = l1_loss(s_img, t_img)
        loss = (1.0 - args.lambda_dssim) * l1_t + args.lambda_dssim * (1.0 - ssim(s_img, t_img))

        # GT only on real view
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
            pbar.set_postfix(loss=float(loss.item()), l1_teacher=float(l1_t.item()), pseudo=int(use_pseudo))

    # ---- save output
    out_dir = args.model_path
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
