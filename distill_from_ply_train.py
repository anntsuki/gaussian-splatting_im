import os, sys, copy, shutil
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
    # exclude DC (l=0)
    return (deg + 1) ** 2 - 1


@torch.no_grad()
def force_truncate_sh(student: GaussianModel, target_sh: int):
    """强制把 features_rest 截断到 target_sh 对应维度（保留低阶系数）"""
    k = sh_rest_dim(target_sh)
    if not hasattr(student, "_features_rest"):
        raise RuntimeError("GaussianModel has no _features_rest")
    fr = student._features_rest
    if fr.shape[-1] < k:
        raise RuntimeError(f"_features_rest dim {fr.shape[-1]} < needed {k}")
    student._features_rest = torch.nn.Parameter(fr[:, :, :k].contiguous())


def set_requires_grad(student: GaussianModel, enable_covariance: bool, enable_opacity: bool):
    # freeze geometry
    if hasattr(student, "_xyz"): student._xyz.requires_grad_(False)
    if hasattr(student, "_scaling"): student._scaling.requires_grad_(enable_covariance)
    if hasattr(student, "_rotation"): student._rotation.requires_grad_(enable_covariance)
    if hasattr(student, "_opacity"): student._opacity.requires_grad_(enable_opacity)

    # train SH only
    if hasattr(student, "_features_dc"): student._features_dc.requires_grad_(True)
    if hasattr(student, "_features_rest"): student._features_rest.requires_grad_(True)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    parser = ArgumentParser("PLY-based SH distillation (no .pth)")

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    # teacher / save
    parser.add_argument("--teacher_model_path", type=str, required=True,
                        help="teacher 模型目录（里面有 cfg_args + point_cloud/iteration_x/point_cloud.ply）")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="从 teacher 的 iteration_x 读取 ply")
    parser.add_argument("--save_iteration", type=int, default=30000,
                        help="把蒸馏后的 ply 保存到 iteration_x（默认 30000，方便你 benchmark 继续用 30000）")

    # distill setup
    parser.add_argument("--new_max_sh", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--lr_sh", type=float, default=1e-2)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)

    # pseudo/augmented view
    parser.add_argument("--augmented_view", action="store_true")
    parser.add_argument("--pv_trans", type=float, default=0.05)
    parser.add_argument("--pv_rot", type=float, default=0.0)

    # freeze toggles
    parser.add_argument("--enable_covariance", action="store_true")
    parser.add_argument("--enable_opacity", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    torch.cuda.set_device(0)

    # dataset & pipe from args (-s, -m, -r 等都由 ModelParams 接管)
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    # background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---------- Load teacher (SH=old) from eval_line2/<scene> ----------
    teacher_ds = copy.deepcopy(dataset)
    teacher_ds.model_path = args.teacher_model_path
    teacher = GaussianModel(teacher_ds.sh_degree)
    teacher_scene = Scene(teacher_ds, teacher, load_iteration=args.iteration, shuffle=False)

    # ---------- Load student from same ply (init = teacher), then down-sh ----------
    student = GaussianModel(teacher_ds.sh_degree)
    _ = Scene(teacher_ds, student, load_iteration=args.iteration, shuffle=False)

    target_sh = args.new_max_sh

    with torch.no_grad():
        # 如果你的分支有 onedownSHdegree，先用它（跟 LightGaussian 逻辑一致）
        if hasattr(student, "onedownSHdegree"):
            try:
                student.max_sh_degree = target_sh
            except Exception:
                pass
            # 多调用几次兜底（有的实现一次只降 1 阶）
            for _ in range(3):
                try:
                    student.onedownSHdegree()
                except Exception:
                    break

        # 再保险：强制截断到 target_sh 对应维度
        force_truncate_sh(student, target_sh)

    # freeze / train flags
    set_requires_grad(student, enable_covariance=args.enable_covariance, enable_opacity=args.enable_opacity)

    # optimizer: SH only
    optim = torch.optim.Adam(
        [{"params": [student._features_dc], "lr": args.lr_sh},
         {"params": [student._features_rest], "lr": args.lr_sh}]
    )

    cams = teacher_scene.getTrainCameras()
    assert len(cams) > 0, "No train cameras loaded"
    use_exp = getattr(dataset, "train_test_exp", False)

    print(f"[INFO] Distill: SH{teacher_ds.sh_degree} -> SH{target_sh}, iters={args.iters}, pseudo={args.augmented_view}")

    pbar = tqdm(range(1, args.iters + 1), desc="Distill SH")
    for it in pbar:
        cam_org = cams[randint(0, len(cams) - 1)]
        cam = copy.deepcopy(cam_org)

        # pseudo-view（跟你贴的 distill_train.py 同款节奏：每 3 次里 2 次用扰动）
        if args.augmented_view and (it % 3 != 0):
            cam = gaussian_poses(cam, mean=0, std_dev_translation=args.pv_trans, std_dev_rotation=args.pv_rot)

        with torch.no_grad():
            t_img = render(cam, teacher, pipe, background, use_trained_exp=use_exp)["render"].detach()

        s_img = render(cam, student, pipe, background, use_trained_exp=use_exp)["render"]

        Ll1 = l1_loss(s_img, t_img)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(s_img, t_img))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if it % 50 == 0:
            pbar.set_postfix(loss=float(loss.item()), l1=float(Ll1.item()))

    # ---------- Save output model folder (cfg_args + exposure.json + point_cloud/iteration_x/point_cloud.ply) ----------
    out_dir = args.model_path
    ensure_dir(out_dir)

    # copy cfg_args（benchmark 脚本靠它）
    for fname in ["cfg_args", "cameras.json", "exposure.json"]:
        src = os.path.join(args.teacher_model_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))

    pc_dir = os.path.join(out_dir, "point_cloud", f"iteration_{args.save_iteration}")
    ensure_dir(pc_dir)
    out_ply = os.path.join(pc_dir, "point_cloud.ply")
    student.save_ply(out_ply)

    print("[DONE] Saved distilled model:")
    print("  Model dir:", out_dir)
    print("  PLY      :", out_ply)


if __name__ == "__main__":
    main()
