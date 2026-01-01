# co_adapt_finetune.py
import os
import sys
from random import randint
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state


def _make_gaussian_model(dataset, opt):
    # 兼容你分支里 GaussianModel(sh_degree, optimizer_type) / GaussianModel(sh_degree) 两种签名
    try:
        return GaussianModel(dataset.sh_degree, opt.optimizer_type)
    except TypeError:
        return GaussianModel(dataset.sh_degree)


def _scale_optimizer_lr(optimizer, lr_scale: float):
    if lr_scale == 1.0:
        return
    for g in optimizer.param_groups:
        if "lr" in g and g["lr"] is not None:
            g["lr"] *= lr_scale


def training_co_adapt(dataset, opt, pipe, load_iter: int, co_iters: int, save_every: int, save_iter: int, lr_scale: float):
    # 让学习率 schedule 的“总迭代数”合理（有些实现会用 opt.iterations 做 max step）
    opt.iterations = int(load_iter + co_iters)

    gaussians = _make_gaussian_model(dataset, opt)
    scene = Scene(dataset, gaussians, load_iteration=load_iter, shuffle=True)

    # 初始化 optimizer（co-adaptation 是“短训恢复”，重新建 optimizer 就行）
    try:
        gaussians.training_setup(opt)
    except AttributeError as e:
        # 如果你分支里 exposure 相关会报错，这里兜底关掉 exposure（大多数 360 配置本来也不需要）
        if "_exposure" in str(e):
            if hasattr(dataset, "train_test_exp"):
                dataset.train_test_exp = False
            if hasattr(opt, "train_test_exp"):
                opt.train_test_exp = False
            gaussians.training_setup(opt)
        else:
            raise

    _scale_optimizer_lr(gaussians.optimizer, lr_scale)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    pbar = tqdm(range(load_iter + 1, opt.iterations + 1), desc=f"Co-adapt from {load_iter} (+{co_iters})")

    for iteration in pbar:
        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack or len(viewpoint_stack) == 0:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染 + photometric loss（论文里就是用原训练视角 photometric loss 做恢复）:contentReference[oaicite:1]{index=1}
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            use_trained_exp=dataset.train_test_exp,
        )
        image = render_pkg["render"]
        gt = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt))

        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix(loss=float(loss.item()), L1=float(Ll1.item()))

        # 定期保存（可选）
        if save_every > 0 and iteration % save_every == 0:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            scene.save(iteration)

    # 最终保存到你指定的 save_iter（默认 load_iter + co_iters）
    final_iter = int(save_iter) if save_iter is not None else int(opt.iterations)
    if final_iter != opt.iterations:
        # 你要一个“固定编号”的保存点，就再存一次
        print(f"\n[FINAL SAVE] Saving Gaussians at iter {final_iter}")
        scene.save(final_iter)
    else:
        print(f"\n[FINAL SAVE] Saving Gaussians at iter {opt.iterations}")
        scene.save(opt.iterations)


@torch.no_grad()
def main():
    parser = ArgumentParser("Co-adaptation (prune recovery) finetune")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--load_iteration", type=int, default=30000, help="which iteration PLY to load from model_path")
    parser.add_argument("--co_iters", type=int, default=5000, help="finetune steps for co-adaptation (paper uses ~5k) :contentReference[oaicite:2]{index=2}")
    parser.add_argument("--save_every", type=int, default=0, help="save every N iters; 0 disables periodic saving")
    parser.add_argument("--save_iteration", type=int, default=None, help="final save iteration tag (default: load_iteration+co_iters)")
    parser.add_argument("--lr_scale", type=float, default=0.1, help="scale down all learning rates for stable short finetune")

    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    load_iter = int(args.load_iteration)
    co_iters = int(args.co_iters)
    save_iter = args.save_iteration if args.save_iteration is not None else (load_iter + co_iters)

    print(f"Model path: {args.model_path}")
    print(f"Source path: {dataset.source_path}")
    print(f"Load iter: {load_iter}  |  Co-iters: {co_iters}  |  Save iter: {save_iter}")
    print("NOTE: Co-adaptation finetunes WITHOUT densification (no clone/split). :contentReference[oaicite:3]{index=3}")

    # 这里必须开启梯度（训练），所以临时关掉 no_grad
    torch.set_grad_enabled(True)
    training_co_adapt(dataset, opt, pipe, load_iter, co_iters, args.save_every, save_iter, args.lr_scale)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
