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
SEARCH_DIR = "eval_line2"
RENDER_SCRIPT = "render.py"
METRICS_SCRIPT = "metrics.py"

# 【创新点2 Pro：频率分层剪枝策略】
# 相比“一刀切”，我们对高频信息砍得更狠，对低频信息保护得更好。
# 这种策略符合 3DGS 的物理特性，论文里非常好吹。
PRUNE_CONFIG = {
    # Band 3 (最精细反光): 砍掉最不重要的 70%
    'band3_ratio': 0.70,
    # Band 2 (普通反光): 砍掉最不重要的 50%
    'band2_ratio': 0.50,
    # Band 1 (基础色彩): 完全保留 (0.0)，防止颜色变灰
    'band1_ratio': 0.00
}


def compress_frequency_logic(input_path, output_path):
    """
    频率分层剪枝逻辑：Band-wise Pruning
    """
    print(f"   -> [Frequency Strategy] Processing {input_path}...")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(input_path)

    # f_rest shape: [N, 15, 3] -> (Band1: 0-2, Band2: 3-7, Band3: 8-14) ??
    # 注意：3DGS存储顺序通常是 DC(0), Rest(1..15)。
    # 在 GaussianModel 内部：
    # _features_dc: [N, 1, 3]
    # _features_rest: [N, 15, 3]
    # 索引对应关系 (SH Degree 3):
    # Band 1 (Degree 1): 3个系数 -> _features_rest[:, 0:3]
    # Band 2 (Degree 2): 5个系数 -> _features_rest[:, 3:8]
    # Band 3 (Degree 3): 7个系数 -> _features_rest[:, 8:15]

    f_rest = gaussians._features_rest
    total_points = f_rest.shape[0]

    # --- 1. 处理 Band 3 (Degree 3) ---
    # 计算 Band 3 能量
    band3_energy = f_rest[:, 8:15, :].abs().mean(dim=(1, 2))  # Shape [N]
    k3 = int(total_points * PRUNE_CONFIG['band3_ratio'])
    threshold3, _ = torch.kthvalue(band3_energy, k3)
    # 生成掩码：能量低于阈值的，把 Band 3 置零
    mask3 = (band3_energy <= threshold3)
    f_rest[mask3, 8:15, :] = 0.0
    print(f"      [Band 3] Pruned {mask3.sum().item()} points ({PRUNE_CONFIG['band3_ratio'] * 100}%).")

    # --- 2. 处理 Band 2 (Degree 2) ---
    # 计算 Band 2 能量
    band2_energy = f_rest[:, 3:8, :].abs().mean(dim=(1, 2))
    k2 = int(total_points * PRUNE_CONFIG['band2_ratio'])
    threshold2, _ = torch.kthvalue(band2_energy, k2)
    mask2 = (band2_energy <= threshold2)
    f_rest[mask2, 3:8, :] = 0.0
    print(f"      [Band 2] Pruned {mask2.sum().item()} points ({PRUNE_CONFIG['band2_ratio'] * 100}%).")

    # --- 3. 处理 Band 1 (Degree 1) - 可选 ---
    if PRUNE_CONFIG['band1_ratio'] > 0:
        band1_energy = f_rest[:, 0:3, :].abs().mean(dim=(1, 2))
        k1 = int(total_points * PRUNE_CONFIG['band1_ratio'])
        threshold1, _ = torch.kthvalue(band1_energy, k1)
        mask1 = (band1_energy <= threshold1)
        f_rest[mask1, 0:3, :] = 0.0
        print(f"      [Band 1] Pruned {mask1.sum().item()} points ({PRUNE_CONFIG['band1_ratio'] * 100}%).")

    # 保存
    gaussians.save_ply(output_path)


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def zip_file(input_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_path, os.path.basename(input_path))


def find_models(root_dir):
    models = []
    abs_root = os.path.abspath(root_dir)
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
    print(f"Found {len(models)} models.")

    for idx, model in enumerate(models):
        print(f"\n[{idx + 1}/{len(models)}] Processing scene: {model['scene_name']}")

        ply_path = model['ply_path']
        backup_path = ply_path + ".backup"

        # 备份
        if not os.path.exists(backup_path):
            shutil.copy2(ply_path, backup_path)
        else:
            shutil.copy2(backup_path, ply_path)

        try:
            # Baseline Zip
            orig_zip_path = ply_path.replace(".ply", "_orig.zip")
            zip_file(backup_path, orig_zip_path)
            orig_zip_size = get_file_size_mb(orig_zip_path)

            # 执行频率分层剪枝
            compress_frequency_logic(backup_path, ply_path)

            # Ours Zip
            ours_zip_path = ply_path.replace(".ply", "_ours.zip")
            zip_file(ply_path, ours_zip_path)
            ours_zip_size = get_file_size_mb(ours_zip_path)

            compression_ratio = (1 - ours_zip_size / orig_zip_size) * 100
            print(f"   -> Storage: {orig_zip_size:.2f}MB -> {ours_zip_size:.2f}MB (Reduced {compression_ratio:.1f}%)")

            # Render & Metrics
            print("   -> Rendering & Evaluating...")
            subprocess.run(f"python {RENDER_SCRIPT} -m \"{model['model_path']}\" --skip_train --quiet", shell=True,
                           check=True)
            subprocess.run(f"python {METRICS_SCRIPT} -m \"{model['model_path']}\"", shell=True, check=True)

            # Read Results
            json_path = os.path.join(model['model_path'], "results.json")
            best_psnr = 0
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for key, val in data.items():
                        if ("ours" in key or "test" in key) and isinstance(val, dict) and 'PSNR' in val:
                            if val['PSNR'] > best_psnr:
                                best_psnr = val['PSNR']

            print(f"   -> Result: PSNR={best_psnr:.2f}")
            results.append({
                "Scene": model['scene_name'],
                "Orig(MB)": round(orig_zip_size, 2),
                "Ours(MB)": round(ours_zip_size, 2),
                "Reduc(%)": round(compression_ratio, 2),
                "PSNR": round(best_psnr, 3)
            })

        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, ply_path)

    # Output CSV
    print("\nFINAL RESULTS (Frequency Strategy)")
    header = ["Scene", "Orig(MB)", "Ours(MB)", "Reduc(%)", "PSNR"]
    print(f"{header[0]:<15} {header[1]:<10} {header[2]:<10} {header[3]:<10} {header[4]:<8}")
    for res in results:
        print(f"{res['Scene']:<15} {res['Orig(MB)']:<10} {res['Ours(MB)']:<10} {res['Reduc(%)']:<10} {res['PSNR']:<8}")


if __name__ == "__main__":
    with torch.no_grad():
        main()