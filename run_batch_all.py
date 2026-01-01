import os
import subprocess
import sys

# ================= 核心配置 =================
# 1. 蒸馏脚本路径
script_name = "distill_from_ply_train.py"

# 2. 输入模型路径 (Teacher: 改进点1 剪枝后的模型)
teacher_root = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_line2"

# 3. 输出模型路径 (Student: 改进点3 蒸馏后的模型)
output_root = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_line2_sh2pv"

# 4. 数据集路径配置 (根据你提供的信息)
path_360 = "/root/autodl-tmp/gaussian-splatting/assets/360_v2"
path_tandt = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/tandt"
path_db = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/db"

# 5. 场景分配列表
scenes_360 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
scenes_tandt = ["train", "truck"]
scenes_db = ["drjohnson", "playroom"]


# ===========================================

def run_distill(scene_name, data_root):
    # 构造具体路径
    dataset_path = os.path.join(data_root, scene_name)
    teacher_path = os.path.join(teacher_root, scene_name)
    model_output_path = os.path.join(output_root, scene_name)

    # 1. 检查 Teacher 模型是否存在 (如果你有些场景没跑剪枝，这步会跳过)
    if not os.path.exists(teacher_path):
        print(f"⚠️  [Skip] Teacher model not found for: {scene_name}")
        return

    # 2. 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"⚠️  [Skip] Dataset not found: {dataset_path}")
        return

    print(f"\n🚀 Processing Scene: {scene_name} (SH=2)")
    print(f"   Data: {dataset_path}")

    # 构造命令 (参数和你之前跑 bicycle 的完全一致)
    cmd = [
        sys.executable, script_name,
        "-s", dataset_path,
        "-m", model_output_path,
        "--teacher_model_path", teacher_path,
        "--iteration", "30000",
        "--save_iteration", "30000",
        "--new_max_sh", "1",  # <--- 想要 Degree 1 这里改成 "1"
        "--iters", "12000",
        "--lr_sh", "3e-3",
        "--lambda_dssim", "0.2",
        "--w_gt", "0.7",
        "--augmented_view",
        "--pseudo_start_ratio", "0.7",
        "--pv_every", "4",
        "--pv_trans", "0.03",
        "--pv_rot", "0.0"
    ]

    # 执行
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Error occurred in scene: {scene_name}")


if __name__ == "__main__":
    # 执行 360 数据集
    print("=== Starting Mip-NeRF 360 Scenes ===")
    for scene in scenes_360:
        run_distill(scene, path_360)

    # 执行 Tanks & Temples 数据集
    print("\n=== Starting Tanks & Temples Scenes ===")
    for scene in scenes_tandt:
        run_distill(scene, path_tandt)

    # 执行 Deep Blending 数据集
    print("\n=== Starting Deep Blending Scenes ===")
    for scene in scenes_db:
        run_distill(scene, path_db)

    print("\n✅ All jobs finished!")