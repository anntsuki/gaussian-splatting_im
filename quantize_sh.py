import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import MiniBatchKMeans  # [NEW] 导入 MiniBatchKMeans 速度更快
import argparse


def quantize_sh_coefficients(input_ply, output_ply, n_clusters=256):
    print(f"Loading model from {input_ply}...")
    plydata = PlyData.read(input_ply)

    properties = [p.name for p in plydata.elements[0].properties]
    sh_rest_names = [p for p in properties if p.startswith("f_rest_")]
    sh_rest_names = sorted(sh_rest_names, key=lambda x: int(x.split('_')[-1]))

    if not sh_rest_names:
        print("No SH coefficients found.")
        return

    sh_data = np.stack([plydata.elements[0][name] for name in sh_rest_names], axis=1)
    num_points = len(sh_data)
    print(f"Total points: {num_points}")

    # --- [优化方案：子采样 + MiniBatchKMeans] ---
    # 1. 即使有 140 万个点，我们只需要 10-20 万个点就能确定很有代表性的聚类中心
    max_samples = 200000
    if num_points > max_samples:
        print(f"Subsampling {max_samples} points for clustering to speed up...")
        indices = np.random.choice(num_points, max_samples, replace=False)
        train_data = sh_data[indices]
    else:
        train_data = sh_data

    print(f"Running MiniBatchKMeans with K={n_clusters}...")
    # MiniBatchKMeans 比普通 KMeans 快得多，适合百万级数据
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1024,  # 每次处理的小批量大小
        n_init=3,  # 运行次数
        random_state=42,
        verbose=1
    )

    # 使用子采样数据拟合模型（训练码本）
    kmeans.fit(train_data)

    # 为所有点分配索引（预测）
    print("Assigning all points to clusters...")
    labels = kmeans.predict(sh_data)
    codebook = kmeans.cluster_centers_
    # ------------------------------------------

    print("Reconstructing quantized SH coefficients...")
    quantized_sh = codebook[labels]

    print("Preparing and saving new PLY data...")
    new_data = plydata.elements[0].data.copy()
    for i, name in enumerate(sh_rest_names):
        new_data[name] = quantized_sh[:, i]

    el = PlyElement.describe(new_data, 'vertex')
    PlyData([el]).write(output_ply)
    print(f"Done! Saved to {output_ply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-k", "--clusters", type=int, default=256)
    args = parser.parse_args()
    quantize_sh_coefficients(args.input, args.output, args.clusters)