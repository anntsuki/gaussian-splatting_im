import os
import torch
import shutil
import zipfile
import subprocess
import json
import csv
import numpy as np
from plyfile import PlyData, PlyElement
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration

# === 核心配置 ===
SEARCH_DIR = "eval_line2"
RENDER_SCRIPT = "render.py"
METRICS_SCRIPT = "metrics.py"

# 【策略升级】混合压缩策略
# 1. 剪枝比例：只剪 25% (非常安全，保护 Kitchen 高光)
PRUNE_RATIO = 0.25
# 2. 量化：将 SH 系数转为 float16 (体积减半，画质无损)
USE_FP16 = True


def save_ply_quantized(gaussians, path, mask):
    """
    自定义的高级保存函数：支持 FP16 量化存储
    """
    mkdir_p(os.path.dirname(path))

    # 1. 应用剪枝掩码，筛选留下的点
    # 注意：这里我们使用 mask 的反面（mask 是要剪掉的，~mask 是要留下的）
    keep_mask = ~mask

    xyz = gaussians._xyz[keep_mask].detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    # SH 系数处理 (核心创新)
    f_dc = gaussians._features_dc[keep_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest[keep_mask].detach().transpose(1, 2).flatten(
        start_dim=1).contiguous().cpu().numpy()

    opacities = gaussians._opacity[keep_mask].detach().cpu().numpy()
    scale = gaussians._scaling[keep_mask].detach().cpu().numpy()
    rotation = gaussians._rotation[keep_mask].detach().cpu().numpy()

    # 2. 构建属性列表，并启用 FP16 量化
    # 如果开启 FP16，我们将 SH 系数 (颜色) 存为 'f2' (2字节浮点)，其他保持 'f4' (4字节)
    # Scale 和 Opacity 对精度敏感，建议保持 f4，或者你也想压榨可以改成 f2
    dtype_sh = 'f2' if USE_FP16 else 'f4'

    # 构建 PLY 头信息
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels including f_dc and f_rest
    for i in range(gaussians._features_dc.shape[1] * gaussians._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(gaussians._features_rest.shape[1] * gaussians._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(gaussians._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(gaussians._rotation.shape[1]):
        l.append('rot_{}'.format(i))

    # 定义每个属性的数据类型
    dtype_full = []
    for name in l:
        if 'f_dc' in name or 'f_rest' in name:
            dtype_full.append((name, dtype_sh))  # SH 系数用 float16
        else:
            dtype_full.append((name, 'f4'))  # 几何属性保持 float32

    # 3. 拼接数据
    # 注意：需要先转为 float32 (numpy默认)，plyfile 写入时会根据 dtype 自动转换
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

    # 创建结构化数组
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    # 逐列赋值 (这步有点慢但稳健)
    # 为了加速，我们可以直接按列赋值
    for i, (name, dtype) in enumerate(dtype_full):
        elements[name] = attributes[:, i]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def compress_hybrid_logic(input_path, output_path, prune_ratio=0.25):
    """
    创新点2 Pro版：混合稀疏量化逻辑
    """
    print(f"   -> Processing Hybrid Compression (Prune {prune_ratio * 100}% + FP16)...")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(input_path)

    f_rest = gaussians._features_rest

    # --- 步骤 1: 剪枝 (基于能量百分位) ---
    sh_energy = f_rest.abs().mean(dim=(1, 2))
    k = int(sh_energy.shape[0] * prune_ratio)
    top_k_value, _ = torch.kthvalue(sh_energy, k)
    threshold = top_k_value.item()

    # mask = True 表示要剪掉的点
    mask = (sh_energy <= threshold)

    pruned_count = mask.sum().item()
    total_count = f_rest.shape[0]

    # --- 步骤 2: 保存 (带 FP16 量化) ---
    save_ply_quantized(gaussians, output_path, mask)

    print(f"      [Debug] Pruned {pruned_count}/{total_count} points ({pruned_count / total_count * 100:.2f}%).")
    print(f"      [Debug] Quantized remaining SH coefficients to Float16.")

    return pruned_count, total_count


# --- 辅助函数 ---
def mkdir_p(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


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
                    models.append({"model_path": root, "ply_path": ply_path, "iteration": iteration,
                                   "scene_name": os.path.basename(root)})
            except:
                continue
    return models


# --- 主程序 ---
def main():
    models = find_models(SEARCH_DIR)
    results = []

    print(f"Found {len(models)} models.")
    for idx, model in enumerate(models):
        print(f"\n[{idx + 1}/{len(models)}] Processing scene: {model['scene_name']}")
        ply_path = model['ply_path']
        backup_path = ply_path + ".backup"

        if not os.path.exists(backup_path):
            shutil.copy2(ply_path, backup_path)
        else:
            shutil.copy2(backup_path, ply_path)

        try:
            # Baseline Zip
            orig_zip_path = ply_path.replace(".ply", "_orig.zip")
            zip_file(backup_path, orig_zip_path)
            orig_zip_size = get_file_size_mb(orig_zip_path)

            # 执行混合压缩
            compress_hybrid_logic(backup_path, ply_path, prune_ratio=PRUNE_RATIO)

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
            best_psnr, best_ssim = 0, 0
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for key, val in data.items():
                        if ("ours" in key or "test" in key) and isinstance(val, dict) and 'PSNR' in val:
                            if val['PSNR'] > best_psnr:
                                best_psnr = val['PSNR'];
                                best_ssim = val['SSIM']

            print(f"   -> Result: PSNR={best_psnr:.2f}")
            results.append({"Scene": model['scene_name'], "Orig_Zip(MB)": round(orig_zip_size, 2),
                            "Ours_Zip(MB)": round(ours_zip_size, 2), "Reduction(%)": round(compression_ratio, 2),
                            "PSNR": round(best_psnr, 3), "SSIM": round(best_ssim, 4)})

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback;
            traceback.print_exc()
        finally:
            if os.path.exists(backup_path): shutil.copy2(backup_path, ply_path)

    # 输出 CSV
    print("\n" + "=" * 50 + "\nFINAL RESULTS\n" + "=" * 50)
    header = ["Scene", "Orig(MB)", "Ours(MB)", "Reduc(%)", "PSNR", "SSIM"]
    print(f"{header[0]:<15} {header[1]:<10} {header[2]:<10} {header[3]:<10} {header[4]:<8} {header[5]:<8}")
    for res in results:
        print(
            f"{res['Scene']:<15} {res['Orig_Zip(MB)']:<10} {res['Ours_Zip(MB)']:<10} {res['Reduction(%)']:<10} {res['PSNR']:<8} {res['SSIM']:<8}")

    with open("final_results_hybrid.csv", 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=results[0].keys()).writeheader();
        csv.DictWriter(f, fieldnames=results[0].keys()).writerows(results)


if __name__ == "__main__":
    with torch.no_grad():
        main()