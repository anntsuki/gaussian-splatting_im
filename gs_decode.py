#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, argparse, shutil, time, subprocess, sys
import numpy as np
from plyfile import PlyData, PlyElement

def find_latest_iter_dir(model_path: str):
    cand = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*"))
    cand = [d for d in cand if os.path.isdir(d)]
    if not cand:
        raise FileNotFoundError(f"Cannot find point_cloud/iteration_* under {model_path}")
    def itnum(d):
        m = re.search(r"iteration_(\d+)", d)
        return int(m.group(1)) if m else -1
    cand.sort(key=itnum)
    return cand[-1], itnum(cand[-1])

def backup_to_temp(ply_path: str, tag: str):
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = ply_path + f".bak_eval_{tag}_{ts}"
    shutil.copy2(ply_path, bak)
    return bak


def write_ply(path: str, pos, nrm, dc, rest, opacity, scale, rot):
    # --- 修改开始：自动填充 SH 系数以满足 GaussianModel 的检查 ---
    # 你的压缩只保留了 SH=1 (9个系数)，但 GaussianModel 通常配置为 SH=3 (45个系数)
    # 我们需要补 0 凑齐 45 个，否则 load_ply 会报错 AssertionError

    current_dim = rest.shape[1]
    # 目标维度：SH3 = 45, SH2 = 24. 默认按标准 3DGS 补齐到 45
    target_sh_degree = 3
    target_dim = 3 * (target_sh_degree + 1) ** 2 - 3  # 45

    if current_dim < target_dim:
        print(f"[WARN] Padding SH coeffs from {current_dim} to {target_dim} to match SH={target_sh_degree} config.")
        padding = np.zeros((rest.shape[0], target_dim - current_dim), dtype=np.float32)
        rest = np.concatenate([rest, padding], axis=1)
    # --- 修改结束 ---

    N = pos.shape[0]
    dtype = [
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ] + [(f"f_rest_{i}", "f4") for i in range(rest.shape[1])] + [
                ("opacity", "f4"),
                ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
                ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
            ]

    arr = np.empty(N, dtype=np.dtype(dtype))
    arr["x"], arr["y"], arr["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    arr["nx"], arr["ny"], arr["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]

    # 这里的循环会自动处理补齐后的所有系数
    for i in range(rest.shape[1]):
        arr[f"f_rest_{i}"] = rest[:, i]

    arr["opacity"] = opacity[:, 0]
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(path)

def dequant_minmax(q: np.ndarray, mn: np.ndarray, mx: np.ndarray, bits: int):
    qmax = (1 << bits) - 1
    return mn + (mx - mn) * (q.astype(np.float32) / qmax)

def dequant_quat(q: np.ndarray, bits: int):
    if bits == 16:
        rot = q.astype(np.float32) / 32767.0
    else:
        rot = q.astype(np.float32) / 127.0
    n = np.linalg.norm(rot, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return rot / n

def decode_npz_to_ply_inplace(npz_path: str, ply_path: str):
    z = np.load(npz_path, allow_pickle=False)

    cb_r = z["cb_r"].astype(np.float32)
    cb_g = z["cb_g"].astype(np.float32)
    cb_b = z["cb_b"].astype(np.float32)
    idx_r = z["idx_r"].astype(np.int64)
    idx_g = z["idx_g"].astype(np.int64)
    idx_b = z["idx_b"].astype(np.int64)

    r = cb_r[idx_r]  # [N,4]
    g = cb_g[idx_g]
    b = cb_b[idx_b]

    dc = np.stack([r[:,0], g[:,0], b[:,0]], axis=1).astype(np.float32)
    rest = np.concatenate([r[:,1:4], g[:,1:4], b[:,1:4]], axis=1).astype(np.float32)  # [N,9]

    pos = z["pos16"].astype(np.float32)
    nrm = z["nrm16"].astype(np.float32) if "nrm16" in z else np.zeros_like(pos)
    opacity = dequant_minmax(z["op_q"], z["op_mn"], z["op_mx"], int(z["op_bits"])).astype(np.float32)
    scale   = dequant_minmax(z["sc_q"], z["sc_mn"], z["sc_mx"], int(z["sc_bits"])).astype(np.float32)
    rot     = dequant_quat(z["rot_q"], int(z["rot_bits"])).astype(np.float32)

    # 写到临时文件，再原子替换
    tmp = ply_path + ".tmp_decode"
    write_ply(tmp, pos, nrm, dc, rest, opacity, scale, rot)
    os.replace(tmp, ply_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", default="cb256")
    ap.add_argument("--bench", default="benchmark_fps.py", help="path to benchmark_fps.py")
    ap.add_argument("--keep_backup", action="store_true", help="keep backup file after restore")

    # 下面这些就是你平时 benchmark_fps.py 用的参数（常用的我都写了）
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_views", type=int, default=50)
    ap.add_argument("--out_json", default=None)

    args, extra = ap.parse_known_args()  # 允许你额外传 benchmark_fps.py 的其他参数

    iter_dir, itnum = find_latest_iter_dir(args.model_path)
    ply_path = os.path.join(iter_dir, "point_cloud.ply")
    npz_path = os.path.join(iter_dir, f"point_cloud.{args.tag}.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path} (did you run gs_encode.py with tag={args.tag}?)")

    # 默认 out_json：同目录下自动生成
    out_json = args.out_json
    if out_json is None:
        out_json = os.path.join(iter_dir, f"bench_{args.tag}.json")

    print(f"[EVAL] latest iter = {itnum}")
    print(f"[EVAL] npz         = {npz_path}")
    print(f"[EVAL] ply         = {ply_path}")
    print(f"[EVAL] bench       = {args.bench}")
    print(f"[EVAL] out_json    = {out_json}")

    bak = backup_to_temp(ply_path, args.tag)
    print(f"[EVAL] backup ->   = {bak}")

    try:
        print("[EVAL] decoding npz -> point_cloud.ply ...")
        decode_npz_to_ply_inplace(npz_path, ply_path)

        cmd = [
            sys.executable, args.bench,
            "-m", args.model_path,
            "-s", args.source_path,
            "--iteration", str(args.iteration),
            "--split", args.split,
            "--n_views", str(args.n_views),
            "--out_json", out_json
        ] + extra

        print("[EVAL] running benchmark:")
        print("       " + " ".join(cmd))
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"[EVAL] benchmark exited with code {ret.returncode}")
            sys.exit(ret.returncode)

        print("[EVAL] benchmark done.")
    finally:
        # 不管 benchmark 成功失败，都恢复
        if os.path.exists(bak):
            os.replace(bak, ply_path)
            print("[EVAL] restored original point_cloud.ply")
            if not args.keep_backup:
                # 已经 restore 了，bak 文件路径被 replace 掉了，不存在了
                pass

if __name__ == "__main__":
    main()
