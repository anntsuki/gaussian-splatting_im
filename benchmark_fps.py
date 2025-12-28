# benchmark_fps.py
import os, json, time
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render

@torch.no_grad()
def main():
    parser = ArgumentParser()
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    # parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--n_views", type=int, default=50, help="how many views to benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out_json", type=str, default="bench.json")
    args = get_combined_args(parser)

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.training_setup(opt)  # 为了保持一致性（虽然后面不训练）

    cams = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    cams = cams[: min(len(cams), args.n_views)]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # warmup
    for i in range(min(args.warmup, len(cams))):
        _ = render(cams[i], gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)["render"]
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for cam in cams:
        start.record()
        _ = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)["render"]
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_ms = float(sum(times_ms) / len(times_ms))
    fps = 1000.0 / avg_ms
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # ply size
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    ply_mb = os.path.getsize(ply_path) / (1024**2) if os.path.exists(ply_path) else None

    # num gaussians
    n_pts = int(gaussians.get_xyz.shape[0])

    out = {
        "model_path": args.model_path,
        "iteration": args.iteration,
        "n_views": len(cams),
        "avg_ms": avg_ms,
        "fps": fps,
        "peak_mem_mb": peak_mem_mb,
        "ply_mb": ply_mb,
        "num_gaussians": n_pts,
    }
    print(json.dumps(out, indent=2))

    with open(os.path.join(args.model_path, args.out_json), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
