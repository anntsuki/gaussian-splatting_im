import os
import torch
import shutil
import zipfile
import subprocess
import json
import csv
from scene.gaussian_model import GaussianModel
from argparse import ArgumentParser
from utils.system_utils import searchForMaxIteration

# === 核心配置 ===
# 你的模型根目录
SEARCH_DIR = "/root/autodl-tmp/gaussian-splatting_feature-pruning/eval_line2"
# 你的渲染和测评脚本路径
RENDER_SCRIPT = "render.py"
METRICS_SCRIPT = "metrics.py"
# 创新点2的阈值 (可调整)
PRUNE_THRESHOLD = 0.005


def compress_sh_logic(input_path, output_path, threshold=0.005):
    """
    创新点2的核心代码：自适应 SH 剪枝
    """
    print(f"   -> Processing SH pruning for {input_path}...")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(input_path)

    # 1. 计算能量
    f_rest = gaussians._features_rest
    scales = gaussians.get_scaling
    max_scales = torch.max(scales, dim=1).values
    sh_energy = f_rest.abs().mean(dim=(1, 2))

    # 2. 生成掩码 (保护微小结构 + 剪枝低频背景)
    mask = (sh_energy < threshold) & (max_scales > 0.001)  # 0.001是保护阈值

    # 3. 置零
    gaussians._features_rest[mask] = 0.0

    # 4. 保存
    gaussians.save_ply(output_path)

    # 返回统计数据
    pruned_count = mask.sum().item()
    total_count = f_rest.shape[0]
    return pruned_count, total_count


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def zip_file(input_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_path, os.path.basename(input_path))


def find_models(root_dir):
    models = []
    # 遍历寻找包含 point_cloud 文件夹的目录
    for root, dirs, files in os.walk(root_dir):
        if "point_cloud" in dirs:
            # 这是一个模型文件夹
            # 寻找最大的 iteration
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
    print(f"Found {len(models)} models in {SEARCH_DIR}")

    for idx, model in enumerate(models):
        print(f"\n[{idx + 1}/{len(models)}] Processing scene: {model['scene_name']}")

        ply_path = model['ply_path']
        backup_path = ply_path + ".backup"

        # 1. 备份原始 PLY (如果还没有备份)
        if not os.path.exists(backup_path):
            shutil.copy2(ply_path, backup_path)
        else:
            # 如果已有备份，说明上次中断了，先恢复以保证从头开始
            shutil.copy2(backup_path, ply_path)

        try:
            # 2. 计算原始 ZIP 大小 (作为 Baseline)
            orig_zip_path = ply_path.replace(".ply", "_orig.zip")
            zip_file(backup_path, orig_zip_path)  # 压缩备份文件作为 baseline
            orig_zip_size = get_file_size_mb(orig_zip_path)

            # 3. 执行创新点2 (SH 剪枝) 并覆盖原始 PLY (为了让 render.py 读取)
            n_pruned, n_total = compress_sh_logic(backup_path, ply_path, threshold=PRUNE_THRESHOLD)

            # 4. 计算处理后的 ZIP 大小 (你的创新点指标)
            ours_zip_path = ply_path.replace(".ply", "_ours.zip")
            zip_file(ply_path, ours_zip_path)
            ours_zip_size = get_file_size_mb(ours_zip_path)

            compression_ratio = (1 - ours_zip_size / orig_zip_size) * 100
            print(f"   -> Storage: {orig_zip_size:.2f}MB -> {ours_zip_size:.2f}MB (Reduced {compression_ratio:.1f}%)")

            # 5. 运行 Render
            print("   -> Rendering...")
            cmd_render = f"python {RENDER_SCRIPT} -m \"{model['model_path']}\" --skip_train --quiet"
            subprocess.run(cmd_render, shell=True, check=True)

            # 6. 运行 Metrics
            print("   -> Calculating Metrics...")
            cmd_metrics = f"python {METRICS_SCRIPT} -m \"{model['model_path']}\""
            subprocess.run(cmd_metrics, shell=True, check=True)

            # 7. 读取 Metrics 结果
            json_path = os.path.join(model['model_path'], "results.json")
            psnr, ssim, lpips = 0, 0, 0
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # 通常 metrics.json key 是 "ours_30000" 或类似
                    for key, val in data.items():
                        if "ours" in key or "test" in key:
                            psnr = val['PSNR']
                            ssim = val['SSIM']
                            lpips = val['LPIPS']
                            break

            print(f"   -> Result: PSNR={psnr:.2f}, Size={ours_zip_size:.2f}MB")

            results.append({
                "Scene": model['scene_name'],
                "Orig_Zip(MB)": round(orig_zip_size, 2),
                "Ours_Zip(MB)": round(ours_zip_size, 2),
                "Reduction(%)": round(compression_ratio, 2),
                "PSNR": round(psnr, 3),
                "SSIM": round(ssim, 4),
                "LPIPS": round(lpips, 4)
            })

        except Exception as e:
            print(f"ERROR processing {model['scene_name']}: {e}")

        finally:
            # 8. 恢复原始文件 (清理现场)
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, ply_path)
                # os.remove(backup_path) # 如果你想保留备份，这行注释掉
                pass

    # === 输出最终 CSV ===
    print("\n" + "=" * 50)
    print("FINAL RESULTS (Copy to your Thesis)")
    print("=" * 50)

    csv_file = "final_results_innovation2.csv"
    keys = results[0].keys() if results else []
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    # 打印到控制台
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