import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os
from pathlib import Path


def prune_sh_and_compress(input_path, output_path):
    print(f"Loading {input_path}...")
    plydata = PlyData.read(input_path)

    vertex = plydata['vertex']

    # 获取属性名列表
    prop_names = [p.name for p in vertex.properties]

    # 加载数据到内存
    data = {}
    for name in prop_names:
        data[name] = np.asarray(vertex[name])

    count = len(data['x'])
    print(f"Original Count: {count}")
    print(f"Original Size: {os.path.getsize(input_path) / 1024 / 1024:.2f} MB")

    # =========================================================
    # Step 1: Morton 排序 (Spatial Sorting)
    # =========================================================
    print("1. Applying Spatial Sorting...")
    # 使用简单的 XYZ 字典序排序 (效果接近 Morton Code，但更快)
    sort_indices = np.lexsort((data['z'], data['y'], data['x']))

    for name in prop_names:
        data[name] = data[name][sort_indices]

    # =========================================================
    # Step 2: SH 阶数剪枝 (SH Level Pruning)
    # =========================================================
    print("2. Pruning SH bands (Degree 2 & 3 -> 0)...")

    # f_rest_0 到 f_rest_8 是 Degree 1 (保留)
    # f_rest_9 到 f_rest_44 是 Degree 2 & 3 (置零)
    pruned_count = 0
    # 3DGS 一般有 45 个 rest 系数 (15 * 3)
    # 我们从第 9 个开始全部置零
    for i in range(9, 45):
        key = f"f_rest_{i}"
        if key in data:
            # 将这些高频系数全部设为 0
            data[key] = np.zeros_like(data[key])
            pruned_count += 1

    print(f"   Zeroed out {pruned_count} high-frequency SH coefficients.")

    # =========================================================
    # Step 3: FP16 量化 (Half-Precision)
    # =========================================================
    print("3. Quantizing to FP16...")
    new_dtype = []
    output_data_arrays = []

    # [修复点]: 直接检查 numpy 数据的类型，而不是检查 ply property 对象
    for name in prop_names:
        arr = data[name]

        # 检查 numpy 数组的类型是否为浮点数 ('f')
        if arr.dtype.kind == 'f':
            new_dtype.append((name, 'f2'))  # 'f2' 就是 float16
            output_data_arrays.append(arr.astype(np.float16))
        else:
            new_dtype.append((name, arr.dtype))
            output_data_arrays.append(arr)

    # 组合数据
    output_data = np.empty(count, dtype=new_dtype)
    for i, name in enumerate(prop_names):
        output_data[name] = output_data_arrays[i]

    # =========================================================
    # 保存
    # =========================================================
    print(f"Saving to {output_path}...")
    el = PlyElement.describe(output_data, 'vertex')
    PlyData([el]).write(output_path)

    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply", help="Path to original ply")
    parser.add_argument("--output_ply", default=None, help="Output path")
    args = parser.parse_args()

    if args.output_ply is None:
        p = Path(args.input_ply)
        # 自动命名
        args.output_ply = str(p.parent / "point_cloud_sh1_fp16.ply")

    prune_sh_and_compress(args.input_ply, args.output_ply)