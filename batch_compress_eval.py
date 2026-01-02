import os
import subprocess
import sys

# ================= 配置区域 =================

# 1. 脚本路径 (假设都在当前目录下，如果不是请修改绝对路径)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ENCODE = os.path.join(CURRENT_DIR, "gs_encode.py")
SCRIPT_DECODE = os.path.join(CURRENT_DIR, "gs_decode.py")
SCRIPT_BENCH = os.path.join(CURRENT_DIR, "benchmark_fps.py")  # 你的测速脚本路径

# 2. 核心路径配置
output_root = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_sh3"

# 数据集路径
path_360 = "/root/autodl-tmp/gaussian-splatting/assets/360_v2"
path_tandt = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/tandt"
path_db = "/root/autodl-tmp/gaussian-splatting/assets/tandt_db/db"

# 3. 任务列表定义
# 格式: (数据集名称, 数据集根目录, [场景列表])
tasks = [
    ("MipNeRF360", path_360, ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]),
    ("TandT", path_tandt, ["train", "truck"]),
    ("DeepBlending", path_db, ["drjohnson", "playroom"]),
]

# 4. 通用参数
TAG = "cb256"  # 压缩包后缀
ITERATION = 30000  # 加载的迭代步数
N_VIEWS = 50  # 测速视点数
SPLIT = "test"  # 测试集 split


# ===========================================

def run_command(cmd, description):
    print(f"\n--- [START] {description} ---")
    print(f"Command: {' '.join(cmd)}")
    try:
        # check=True 会在命令报错时抛出异常停止当前场景，但不影响下一个场景
        subprocess.run(cmd, check=True)
        print(f"--- [SUCCESS] {description} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"!!! [ERROR] {description} failed with code {e.returncode} !!!\n")
        return False
    return True


def main():
    print(f"Batch script started. Processing {sum(len(t[2]) for t in tasks)} scenes...")

    for dataset_name, dataset_path, scenes in tasks:
        print(f"t>>> Entering Dataset: {dataset_name}")

        for scene in scenes:
            print(f"t> Processing Scene: {scene}")

            # 拼接路径
            # 模型路径: output_root / scene (例如 .../eval_sh3/bicycle)
            model_path = os.path.join(output_root, scene)
            # 源数据路径: dataset_path / scene (例如 .../360_v2/bicycle)
            source_path = os.path.join(dataset_path, scene)

            # 检查模型是否存在
            if not os.path.exists(model_path):
                print(f"!!! Model path not found: {model_path}. Skipping.")
                continue

            # ---------------------------
            # 步骤 1: Encode (压缩)
            # ---------------------------
            # 命令: python gs_encode.py --model_path X --tag cb256 --K 256 --sh_degree 1 ...
            # 注意：gs_encode 其实不需要 --sh_degree，它根据文件读。但为了保持一致性可以传通用参
            cmd_encode = [
                sys.executable, SCRIPT_ENCODE,
                "--model_path", model_path,
                "--tag", TAG,
                "--K", "4096",  # 码本大小
                "--sample", "200000",  # 采样数
                "--no_morton"  # 如果你想用 morton 排序就去掉这行，不去掉就是原来的逻辑
                # 如果 gs_encode.py 不支持 --no_morton 这种 flag 形式请自行调整，
                # 但根据你给的代码，默认是启用 morton 的 (代码逻辑是 args.no_morton 为 True 才关闭)
                # 所以这里不传 --no_morton 就是启用排序。
            ]
            # 这里的 cmd_encode 如果你的代码里有 --no_morton 且你想启用排序，就不要加那个参数。
            # 你给的代码里默认是开启排序的 (只要不传 --no_morton)。
            # 修正 cmd_encode: 移除 --no_morton 以启用排序获得更小体积
            cmd_encode = [
                sys.executable, SCRIPT_ENCODE,
                "--model_path", model_path,
                "--tag", TAG,
                "--K", "4096"
            ]

            if not run_command(cmd_encode, f"Encoding {scene}"):
                continue  # 如果压缩失败，就不测了，跳下一个

            # ---------------------------
            # 步骤 2: Decode & Benchmark
            # ---------------------------
            # 命令: python gs_decode.py --model_path X --tag cb256 --bench ... -s Y ... --sh_degree 1
            cmd_decode = [
                sys.executable, SCRIPT_DECODE,
                "--model_path", model_path,
                "--tag", TAG,
                "--bench", SCRIPT_BENCH,
                # 下面是传给 benchmark_fps.py 的参数
                "-s", source_path,
                "--iteration", str(ITERATION),
                "--split", SPLIT,
                "--n_views", str(N_VIEWS),
                "--sh_degree", "1"  # <--- 关键！解决报错并提速
            ]

            run_command(cmd_decode, f"Decoding & Benchmarking {scene}")

    print("\nAll batch tasks finished.")


if __name__ == "__main__":
    main()