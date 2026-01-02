#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, argparse, shutil, time
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

def backup_file(path: str):
    if not os.path.exists(path):
        return None
    # 如果已有 .bak，就加时间戳避免覆盖
    base = path + ".bak"
    if not os.path.exists(base):
        shutil.copy2(path, base)
        return base
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = f"{base}.{ts}"
    shutil.copy2(path, bak)
    return bak

def write_ply(path: str, pos, nrm, dc, rest, opacity, scale, rot):
    N = pos.shape[0]
    dtype = [
        ("x","f4"),("y","f4"),("z","f4"),
        ("nx","f4"),("ny","f4"),("nz","f4"),
        ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
    ] + [(f"f_rest_{i}","f4") for i in range(rest.shape[1])] + [
        ("opacity","f4"),
        ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
        ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
    ]
    arr = np.empty(N, dtype=np.dtype(dtype))
    arr["x"], arr["y"], arr["z"] = pos[:,0], pos[:,1], pos[:,2]
    arr["nx"], arr["ny"], arr["nz"] = nrm[:,0], nrm[:,1], nrm[:,2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc[:,0], dc[:,1], dc[:,2]
    for i in range(rest.shape[1]):
        arr[f"f_rest_{i}"] = rest[:, i]
    arr["opacity"] = opacity[:,0]
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scale[:,0], scale[:,1], scale[:,2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rot[:,0], rot[:,1], rot[:,2], rot[:,3]
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", default="cb256")
    ap.add_argument("--npz", default=None, help="optional explicit npz path (otherwise auto: latest_iter/point_cloud.<tag>.npz)")
    ap.add_argument("--no_backup", action="store_true")
    args = ap.parse_args()

    iter_dir, itnum = find_latest_iter_dir(args.model_path)
    print(f"[DEC] latest iter = {itnum}")
    print(f"[DEC] iter dir    = {iter_dir}")

    npz_path = args.npz if args.npz else os.path.join(iter_dir, f"point_cloud.{args.tag}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Cannot find npz: {npz_path}")

    ply_path = os.path.join(iter_dir, "point_cloud.ply")
    print(f"[DEC] use npz     = {npz_path}")
    print(f"[DEC] target ply  = {ply_path}")

    if not args.no_backup:
        bak = backup_file(ply_path)
        if bak:
            print(f"[DEC] backup ply = {bak}")
        else:
            print("[DEC] backup ply = (no original ply found)")

    z = np.load(npz_path, allow_pickle=False)
    K = int(z["K"])

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

    write_ply(ply_path, pos, nrm, dc, rest, opacity, scale, rot)
    print("[DEC] wrote ply   = OK (overwritten point_cloud.ply)")

if __name__ == "__main__":
    main()
