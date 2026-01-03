import os, re, glob, subprocess, sys

# ===== 你要改的配置 =====
output_root_in  = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_sh2pv"
output_root_out = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_sh2pvfine"

path_360   = "/root/autodl-tmp/gaussian-splatting/assets/360_v2"
path_tandt = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/tandt"
path_db    = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/db"

scenes_360   = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
scenes_tandt = ["train", "truck"]
scenes_db    = ["drjohnson", "playroom"]

# 微调步数：按需改（建议 2000~8000 之间试）
FINETUNE_ITERS = 4000
# =======================

REPO = "/root/autodl-tmp/gaussian-splatting_feature-pruning"
FINETUNE_PY = os.path.join(REPO, "finetune.py")

def latest_iter(model_dir: str) -> int:
    it_dirs = glob.glob(os.path.join(model_dir, "point_cloud", "iteration_*"))
    iters = []
    for d in it_dirs:
        m = re.search(r"iteration_(\d+)$", d)
        if m:
            iters.append(int(m.group(1)))
    if not iters:
        raise RuntimeError(f"No iteration_* found under: {model_dir}/point_cloud/")
    return max(iters)

def read_sh_degree(model_dir: str, default: int = 2) -> int:
    cfg = os.path.join(model_dir, "cfg_args")
    if not os.path.exists(cfg):
        return default
    txt = open(cfg, "r", encoding="utf-8", errors="ignore").read()
    # 兼容 "sh_degree=3" 或 "sh_degree 3" 之类
    m = re.search(r"sh_degree\s*=?\s*(\d+)", txt)
    return int(m.group(1)) if m else default

def run_one(scene: str, data_root: str):
    src_model = os.path.join(output_root_in, scene)
    if not os.path.isdir(src_model):
        print(f"[SKIP] model dir not found: {src_model}")
        return

    it = latest_iter(src_model)
    start_ply = os.path.join(src_model, "point_cloud", f"iteration_{it}", "point_cloud.ply")
    if not os.path.exists(start_ply):
        print(f"[SKIP] start ply not found: {start_ply}")
        return

    sh = read_sh_degree(src_model, default=2)
    source_path = os.path.join(data_root, scene)
    out_model = os.path.join(output_root_out, scene)

    cmd = [
        sys.executable, FINETUNE_PY,
        "-s", source_path,
        "-m", out_model,
        "--start_checkpoint", start_ply,
        "--iterations", str(FINETUNE_ITERS),
        "--sh_degree", 1,
        # 让中间也存一次（不想存可删）
    ]

    print("\n" + "="*90)
    print(f"[RUN] scene={scene}  sh_degree={sh}  iter_from={it}")
    print("CMD:", " ".join(cmd))
    print("="*90 + "\n")

    subprocess.run(cmd, check=True, cwd=REPO)

def main():
    os.makedirs(output_root_out, exist_ok=True)

    for s in scenes_360:
        run_one(s, path_360)
    for s in scenes_tandt:
        run_one(s, path_tandt)
    for s in scenes_db:
        run_one(s, path_db)

    print("\n[DONE] All finetune jobs finished.\n")

if __name__ == "__main__":
    main()
