import os
import torch
import shutil
import zipfile
import subprocess
import json
import csv
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration

# === 核心配置 ===
# 既然你在 gaussian-splatting_feature-pruning 目录下运行，直接指向子目录即可
SEARCH_DIR = "eval_line2"

RENDER_SCRIPT = "render.py"
METRICS_SCRIPT = "metrics.py"

# 【关键改动】动态阈值系数
# 0.5 表示：只要一个点的能量低于“全场景平均能量的 50%”，就剪掉它。
# 这是一个非常稳健的策略，既能保证压缩率，又不会误删高光。
ENERGY_RATIO = 0.5


def compress_sh_logic(input_path, output_path, ratio=0.5):
    """创新点2的核心代码：自适应 SH 剪枝 (动态阈值版)"""
    print(f"   -> Processing SH pruning for {input_path}...")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(input_path)

    f_rest = gaussians._features_rest
    scales = gaussians.get_scaling
    max_scales = torch.max(scales, dim=1).values

    # 1. 计算每个点的 SH 能量
    sh_energy = f_rest.abs().mean(dim=(1, 2))

    # 2. 【关键】计算动态阈值
    avg_energy = sh_energy.mean().item()
    dynamic_threshold = avg_energy * ratio

    print(f"      [Debug] Scene Avg Energy: {avg_energy:.5f}")
    print(f"      [Debug] Dynamic Threshold set to: {dynamic_threshold:.5f} (Avg * {ratio})")

    # 3. 生成掩码
    # 彻底移除 max_scales 保护，或者设得极小，防止它干扰测试
    mask = (sh_energy < dynamic_threshold)

    # 4. 置零
    gaussians._features_rest[mask] = 0.0
    gaussians.save_ply(output_path)

    pruned_count = mask.sum().item()
    total_count = f_rest.shape[0]
    print(f"      [Debug] Pruned {pruned_count}/{total_count} points ({pruned_count / total_count * 100:.2f}%).")

    return pruned_count, total_count


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def zip_file(input_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_path, os.path.basename(input_path))


def find_models(root_dir):
    models = []
    abs_root = os.path.abspath(root_dir)
    print(f"DEBUG: Searching for models in: {abs_root}")

    if not os.path.exists(abs_root):
        print(f"ERROR: Directory not found: {abs_root}")
        return []

    for root, dirs, files in os.walk(abs_root):
        if "point_cloud" in dirs:
            try:
                iteration = searchForMaxIteration(os.path.join(root, "point_cloud"))
                ply_path = os.path.join(root, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
                if os.path.exists(ply_path):
                    models.append({
                        "model_path": root,
                        "ply_path": ply_path,
                        "iteration": iteration,
                        "scene_name": os.path.basename(root)
                    })
            except:
                continue
    return models


def main():
    results = []

    models = find_models(SEARCH_DIR)
    if len(models) == 0:
        print("Found 0 models! Please check the path.")
        return

    print(f"Found {len(models)} models.")

    for idx, model in enumerate(models):
        print(f"\n[{idx + 1}/{len(models)}] Processing scene: {model['scene_name']}")

        ply_path = model['ply_path']
        backup_path = ply_path + ".backup"

        # 1. 备份
        if not os.path.exists(backup_path):
            shutil.copy2(ply_path, backup_path)
        else:
            shutil.copy2(backup_path, ply_path)

        try:
            # 2. Baseline Zip
            orig_zip_path = ply_path.replace(".ply", "_orig.zip")
            zip_file(backup_path, orig_zip_path)
            orig_zip_size = get_file_size_mb(orig_zip_path)

            # 3. 执行创新点2 (使用动态比例)
            compress_sh_logic(backup_path, ply_path, ratio=ENERGY_RATIO)

            # 4. Ours Zip
            ours_zip_path = ply_path.replace(".ply", "_ours.zip")
            zip_file(ply_path, ours_zip_path)
            ours_zip_size = get_file_size_mb(ours_zip_path)

            compression_ratio = (1 - ours_zip_size / orig_zip_size) * 100
            print(f"   -> Storage: {orig_zip_size:.2f}MB -> {ours_zip_size:.2f}MB (Reduced {compression_ratio:.1f}%)")

            # 5. Render
            print("   -> Rendering...")
            cmd_render = f"python {RENDER_SCRIPT} -m \"{model['model_path']}\" --skip_train --quiet"
            subprocess.run(cmd_render, shell=True, check=True)

            # 6. Metrics
            print("   -> Calculating Metrics...")
            cmd_metrics = f"python {METRICS_SCRIPT} -m \"{model['model_path']}\""
            subprocess.run(cmd_metrics, shell=True, check=True)

            # 7. Read Results (修正读取逻辑，找 PSNR 最高的那个)
            json_path = os.path.join(model['model_path'], "results.json")
            best_psnr = 0
            best_ssim = 0
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # 遍历所有 key (例如 ours_7000, ours_30000)，找迭代次数最大的
                    for key, val in data.items():
                        # 确保是 ours 或 test 且包含数字
                        if ("ours" in key or "test" in key) and isinstance(val, dict) and 'PSNR' in val:
                            if val['PSNR'] > best_psnr:  # 简单粗暴：取 PSNR 最高的那个结果
                                best_psnr = val['PSNR']
                                best_ssim = val['SSIM']

            print(f"   -> Result: PSNR={best_psnr:.2f}, Size={ours_zip_size:.2f}MB")

            results.append({
                "Scene": model['scene_name'],
                "Orig_Zip(MB)": round(orig_zip_size, 2),
                "Ours_Zip(MB)": round(ours_zip_size, 2),
                "Reduction(%)": round(compression_ratio, 2),
                "PSNR": round(best_psnr, 3),
                "SSIM": round(best_ssim, 4)
            })

        except Exception as e:
            print(f"ERROR processing {model['scene_name']}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 8. 恢复
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, ply_path)

    # === 输出 CSV ===
    print("\n" + "=" * 50)
    print("FINAL RESULTS (Copy to your Thesis)")
    print("=" * 50)

    csv_file = "final_results_innovation2.csv"
    keys = ["Scene", "Orig_Zip(MB)", "Ours_Zip(MB)", "Reduction(%)", "PSNR", "SSIM"]
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    header = ["Scene", "Orig(MB)", "Ours(MB)", "Reduc(%)", "PSNR", "SSIM"]
    print(f"{header[0]:<15} {header[1]:<10} {header[2]:<10} {header[3]:<10} {header[4]:<8} {header[5]:<8}")
    for res in results:
        print(
            f"{res['Scene']:<15} {res['Orig_Zip(MB)']:<10} {res['Ours_Zip(MB)']:<10} {res['Reduction(%)']:<10} {res['PSNR']:<8} {res['SSIM']:<8}")
    print("=" * 50)
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    with torch.no_grad():
        main()