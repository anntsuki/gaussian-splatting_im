# run_all_coadapt_final.py
import os
import subprocess

# ================= 1. 路径配置 =================

# 【关键】请修改为存放“剪枝后模型”的根目录
# 这个目录下应该有 bicycle, garden, train, truck, drjohnson 等子文件夹
MODELS_ROOT = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_line2"

# 数据集路径
PATH_360 = "/root/autodl-tmp/gaussian-splatting/assets/360_v2"
PATH_TANDT = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/tandt"
PATH_DB = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/db"

# ================= 2. 场景定义 =================

# 字典映射：场景名 -> 数据集路径
SCENE_MAPPING = {}

# Mip-NeRF 360
for s in ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]:
    SCENE_MAPPING[s] = os.path.join(PATH_360, s)

# Tanks & Temples
for s in ["train", "truck"]:
    SCENE_MAPPING[s] = os.path.join(PATH_TANDT, s)

# Deep Blending
for s in ["drjohnson", "playroom"]:
    SCENE_MAPPING[s] = os.path.join(PATH_DB, s)

# ================= 3. 运行参数 =================

LOAD_ITER = 30000  # 加载剪枝后的模型
CO_ITERS = 5000  # 跑 5000 步微调
SAVE_ITER = 35000  # 保存为 35000
LR_SCALE = 0.1  # 学习率降低
SH_DEGREE = 3  # 【重要】保持 3 阶 (因为这是剪枝后的恢复，还没做SH降维)


# ================= 4. 主逻辑 =================

def main():
    print(f"🚀 Starting Batch Co-adaptation on {len(SCENE_MAPPING)} scenes...")

    # 遍历所有场景
    for scene_name, source_path in SCENE_MAPPING.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {scene_name}")
        print(f"Dataset:  {source_path}")
        print(f"{'=' * 60}")

        # 对应的模型路径
        model_path = os.path.join(MODELS_ROOT, scene_name)

        # 1. 检查路径是否存在
        if not os.path.exists(source_path):
            print(f"❌ [SKIP] Source path missing: {source_path}")
            continue
        if not os.path.exists(model_path):
            print(f"❌ [SKIP] Model path missing: {model_path}")
            continue

        # 2. 检查是否有 30000 步的模型
        ply_30k = os.path.join(model_path, "point_cloud", f"iteration_{LOAD_ITER}", "point_cloud.ply")
        if not os.path.exists(ply_30k):
            print(f"❌ [SKIP] Iteration {LOAD_ITER} PLY missing in: {model_path}")
            continue

        # 3. 构建命令
        cmd = [
            "python", "co_adapt_finetune.py",
            "-s", source_path,
            "-m", model_path,
            "--load_iteration", str(LOAD_ITER),
            "--co_iters", str(CO_ITERS),
            "--save_iteration", str(SAVE_ITER),
            "--lr_scale", str(LR_SCALE),
            "--sh_degree", str(SH_DEGREE)
        ]

        # 4. 执行
        try:
            print(f"▶️ Executing Co-adaptation...")
            # 这里的 subprocess 会把输出打印到当前终端，方便你看进度
            subprocess.run(cmd, check=True)
            print(f"✅ {scene_name} Done!")
        except subprocess.CalledProcessError as e:
            print(f"❌ {scene_name} Failed with error code {e.returncode}")

    print("\n🎉 All tasks finished!")


if __name__ == "__main__":
    main()