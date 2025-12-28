import numpy as np
import os
import struct
from plyfile import PlyData


def compress_model(input_ply, output_bin):
    print(f"[*] 正在读取量化模型: {input_ply}")
    plydata = PlyData.read(input_ply)
    vertex = plydata['vertex']
    num_points = len(vertex.data)

    # 1. 提取 SH 系数并构建码本 (Codebook)
    print("[*] 正在提取 SH 码本和索引...")
    sh_names = [f"f_rest_{i}" for i in range(45)]
    sh_data = np.stack([vertex[n] for n in sh_names], axis=1).astype(np.float32)

    # 利用 numpy 寻找唯一的 256 个聚类中心
    unique_sh, indices = np.unique(sh_data, axis=0, return_inverse=True)

    # 2. 准备几何与颜色参数 (转为 float16 减少 50% 体积)
    print("[*] 正在执行半精度 (Float16) 转换...")
    float16_attrs = [
        'x', 'y', 'z', 'opacity',
        'f_dc_0', 'f_dc_1', 'f_dc_2',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3'
    ]

    # 开始写入自定义二进制格式
    with open(output_bin, 'wb') as f:
        # 写入文件头: 点数 (int), 码本大小 (int)
        f.write(struct.pack('ii', num_points, len(unique_sh)))

        # 写入码本数据 (SH 聚类中心)
        unique_sh.tofile(f)

        # 写入每个点的 SH 索引 (uint8, 关键压缩步)
        indices.astype(np.uint8).tofile(f)

        # 按顺序写入几何/基础颜色数据 (float16)
        for attr in float16_attrs:
            if attr in vertex:
                vertex[attr].astype(np.float16).tofile(f)

    # 计算对比数据用于论文
    orig_size = os.path.getsize(input_ply) / (1024 * 1024)
    comp_size = os.path.getsize(output_bin) / (1024 * 1024)
    print("-" * 30)
    print(f"创新点 3 压缩结果:")
    print(f"原始 PLY 体积: {orig_size:.2f} MB")
    print(f"压缩后 BIN 体积: {comp_size:.2f} MB")
    print(f"实际物理压缩比: {orig_size / comp_size:.2f}x")
    print("-" * 30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="量化后的 .ply 文件")
    parser.add_argument("-o", "--output", required=True, help="输出的 .bin 压缩文件")
    args = parser.parse_args()
    compress_model(args.input, args.output)