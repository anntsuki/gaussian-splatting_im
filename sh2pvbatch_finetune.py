import os, re, glob, json, subprocess, sys
from datetime import datetime
input_root  = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_sh2pv"
output_root = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_sh2pvfine"

path_360   = "/root/autodl-tmp/gaussian-splatting/assets/360_v2"
path_tandt = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/tandt"
path_db    = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/db"

scenes_360   = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
scenes_tandt = ["train", "truck"]
scenes_db    = ["drjohnson", "playroom"]
# ======================

REPO = "/root/autodl-tmp/gaussian-splatting_feature-pruning"
FINETUNE_PY = os.path.join(REPO, "finetune.py")
BENCH_PY    = os.path.join(REPO, "benchmark_fps.py")

# 你可以改这些
FINETUNE_ITERS = 5000          # 微调迭代数（常用 2000~8000）
N_VIEWS = 50                   # benchmark 视角上限（会被 test set 实际数量截断）
SPLIT = "test"
FORCE = True                  # True: 即使输出已存在也重跑；False: 有结果就跳过微调只测评

# 防碎片（可选但推荐）
ENV = os.environ.copy()
ENV["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"

def latest_iter(model_dir: str) -> int:
    it_dirs = glob.glob(os.path.join(model_dir, "point_cloud", "iteration_*"))
    iters = []
    for d in it_dirs:
        m = re.search(r"iteration_(\d+)$", d)
        if m: iters.append(int(m.group(1)))
    if not iters:
        raise RuntimeError(f"No iteration_* under {model_dir}/point_cloud")
    return max(iters)

def pick_ply(model_dir: str, it: int) -> str:
    ply = os.path.join(model_dir, "point_cloud", f"iteration_{it}", "point_cloud.ply")
    if not os.path.exists(ply):
        raise RuntimeError(f"Missing ply: {ply}")
    return ply

def read_sh_degree(model_dir: str, default: int = 2) -> int:
    cfg = os.path.join(model_dir, "cfg_args")
    if not os.path.exists(cfg):
        return default
    txt = open(cfg, "r", encoding="utf-8", errors="ignore").read()
    m = re.search(r"sh_degree\s*=?\s*(\d+)", txt)
    return int(m.group(1)) if m else default

def run_cmd(cmd, cwd=REPO):
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=ENV)

def finetune_scene(scene: str, data_root: str) -> str:
    src_model = os.path.join(input_root, scene)
    if not os.path.isdir(src_model):
        print(f"[SKIP] input model not found: {src_model}")
        return ""

    src_it = latest_iter(src_model)
    start_ply = pick_ply(src_model, src_it)
    sh = read_sh_degree(src_model, default=2)

    source_path = os.path.join(data_root, scene)
    out_model = os.path.join(output_root, scene)

    # 如果已经有输出且不强制重跑：跳过微调
    if (not FORCE) and os.path.isdir(out_model) and glob.glob(os.path.join(out_model, "point_cloud", "iteration_*")):
        print(f"[OK] output exists, skip finetune: {out_model}")
        return out_model

    os.makedirs(out_model, exist_ok=True)

    cmd = [
        sys.executable, FINETUNE_PY,
        "-s", source_path,
        "-m", out_model,
        "--start_checkpoint", start_ply,
        "--iterations", str(FINETUNE_ITERS),
        "--sh_degree", str(sh),
        "--save_iterations", str(FINETUNE_ITERS),
        "--test_iterations", str(FINETUNE_ITERS),
        "--quiet",
    ]
    print("\n" + "="*110)
    print(f"[FINETUNE] scene={scene}  sh_degree={sh}  from={src_model} (iter {src_it})")
    print("="*110)
    run_cmd(cmd)
    return out_model

def benchmark_scene(scene: str, data_root: str, model_dir: str) -> dict:
    if not model_dir:
        return {}

    it = latest_iter(model_dir)     # 微调后真实保存到哪个 iteration，就用哪个
    sh = read_sh_degree(model_dir, default=2)
    source_path = os.path.join(data_root, scene)

    out_dir = os.path.join(model_dir, "bench")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"bench_finetune_iter{it}.json")

    cmd = [
        sys.executable, BENCH_PY,
        "-m", model_dir,
        "-s", source_path,
        "--iteration", str(it),
        "--split", SPLIT,
        "--n_views", str(N_VIEWS),
        "--out_json", out_json,
        "--sh_degree", str(sh),
    ]
    print("\n" + "-"*110)
    print(f"[BENCH] scene={scene}  sh_degree={sh}  model={model_dir}  iter={it}")
    print("-"*110)
    run_cmd(cmd)

    # 读回结果，汇总
    with open(out_json, "r") as f:
        res = json.load(f)
    return res

def run_dataset(tag, data_root, scenes):
    results = []
    for s in scenes:
        out_model = finetune_scene(s, data_root)
        res = benchmark_scene(s, data_root, out_model)
        if res:
            res["dataset_tag"] = tag
            res["scene_name"] = s
            results.append(res)
    return results

def main():
    os.makedirs(output_root, exist_ok=True)
    all_results = []
    all_results += run_dataset("360_v2", path_360, scenes_360)
    all_results += run_dataset("tandt",  path_tandt, scenes_tandt)
    all_results += run_dataset("db",     path_db, scenes_db)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_root, f"summary_finetune_bench_{stamp}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[DONE] All finetune+bench finished.")
    print("Summary JSON:", summary_path)

if __name__ == "__main__":
    main()