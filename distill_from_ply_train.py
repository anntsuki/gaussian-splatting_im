import os, sys, copy
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
    return (deg + 1) ** 2 - 1

@torch.no_grad()
def truncate_sh_to(student: GaussianModel, target_sh: int):
    """把 student 的 SH rest 系数截断到 target_sh 对应维度（保留低阶系数）"""
    k = sh_rest_dim(target_sh)
    if hasattr(student, "_features_rest"):
        if student._features_rest.shape[-1] >= k:
            student._features_rest = torch.nn.Parameter(student._features_rest[:, :, :k].contiguous())
        else:
            raise RuntimeError(f"features_rest dim too small: {student._features_rest.shape[-1]} < {k}")
    else:
        raise RuntimeError("GaussianModel missing _features_rest")

def freeze_non_sh(student: GaussianModel, enable_covariance: bool, enable_opacity: bool):
    # geometry
    if hasattr(student, "_xyz"): student._xyz.requires_grad_(False)
    if hasattr(student, "_scaling"): student._scaling.requires_grad_(enable_covariance)  # covariance off => freeze
    if hasattr(student, "_rotation"): student._rotation.requires_grad_(enable_covariance)
    if hasattr(student, "_opacity"): student._opacity.requires_grad_(enable_opacity)

    # SH
    student._features_dc.requires_grad_(True)
    student._features_rest.requires_grad_(True)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    parser = ArgumentParser("PLY-based SH distillation (no .pth)")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--teacher_model_path", type=str, required=True,
                        help="teacher 模型目录（里面有 point_cloud/iteration_x/point_cloud.ply）")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--save_iteration", type=int, default=30000)

    parser.add_argument("--new_max_sh", type=int, default=2, choices=[1,2,3])
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--lr_sh", type=float, default=1e-2)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)

    parser.add_argument("--augmented_view", action="store_true")
    parser.add_argument("--pv_trans", type=float, default=0.05)
    parser.add_argument("--pv_rot", type=float, default=0.0)

    parser.add_argument("--enable_covariance", action="store_true")
    parser.add_argument("--enable_opacity", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    # background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---- load teacher from PLY
    teacher_ds = copy.deepcopy(dataset)
    teacher_ds.model_path = args.teacher_model_path
    teacher = GaussianModel(teacher_ds.sh_degree)
    teacher_scene = Scene(teacher_ds, teacher, load_iteration=args.iteration, shuffle=False)
    teacher.eval()

    # ---- load student from same PLY, then down-sh to new_max_sh
    student = GaussianModel(teacher_ds.sh_degree)
    _ = Scene(teacher_ds, student, load_iteration=args.iteration, shuffle=False)  # load same ply into student

    # 降到 SH=2（先调用 onedownSHdegree，如果你分支有；没有就直接截断）
    target_sh = args.new_max_sh
    with torch.no_grad():
        if hasattr(student, "onedownSHdegree"):
            student.max_sh_degree = target_sh
            # 多调用几次兜底（有的实现一次只降 1 阶）
            for _ in range(3):
                try:
                    student.onedownSHdegree()
                except Exception:
                    break
        # 再保险：强制截断到 target_sh 对应维度
        truncate_sh_to(student, target_sh)

    # 冻结非 SH
    freeze_non_sh(student, enable_covariance=args.enable_covariance, enable_opacity=args.enable_opacity)

    # optimizer（只训 SH）
    optim = torch.optim.Adam(
        [{"params": [student._features_dc], "lr": args.lr_sh},
         {"params": [student._features_rest], "lr": args.lr_sh}]
    )

    # cams
    cams = teacher_scene.getTrainCameras()
    assert len(cams) > 0

    use_exp = getattr(dataset, "train_test_exp", False)

    pbar = tqdm(range(1, args.iters + 1), desc="Distill SH")
    for it in pbar:
        cam_org = cams[randint(0, len(cams) - 1)]
        cam = copy.deepcopy(cam_org)

        if args.augmented_view and (it % 3 != 0):
            cam = gaussian_poses(cam, mean=0, std_dev_translation=args.pv_trans, std_dev_rotation=args.pv_rot)

        # teacher target
        with torch.no_grad():
            t_img = render(cam, teacher, pipe, background, use_trained_exp=use_exp)["render"].detach()

        # student
        s_img = render(cam, student, pipe, background, use_trained_exp=use_exp)["render"]

        Ll1 = l1_loss(s_img, t_img)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(s_img, t_img))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if it % 50 == 0:
            pbar.set_postfix(loss=float(loss.item()), l1=float(Ll1.item()))

    # ---- save distilled ply (保持你评测脚本兼容：cfg_args + point_cloud/iteration_x/point_cloud.ply)
    out_dir = args.model_path
    ensure_dir(out_dir)
    # copy cfg_args for evaluation scripts
    cfg_src = os.path.join(args.teacher_model_path, "cfg_args")
    if os.path.exists(cfg_src):
        import shutil
        shutil.copy(cfg_src, os.path.join(out_dir, "cfg_args"))

    pc_dir = os.path.join(out_dir, "point_cloud", f"iteration_{args.save_iteration}")
    ensure_dir(pc_dir)
    out_ply = os.path.join(pc_dir, "point_cloud.ply")
    student.save_ply(out_ply)

    print("Saved distilled model to:", out_dir)
    print("PLY:", out_ply)

if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
