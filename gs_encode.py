#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, argparse
import numpy as np
import torch
from plyfile import PlyData

def find_latest_iter_ply(model_path: str):
    cand = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply"))
    if not cand:
        raise FileNotFoundError(f"Cannot find point_cloud/iteration_*/point_cloud.ply under {model_path}")
    def itnum(p):
        m = re.search(r"iteration_(\d+)", p)
        return int(m.group(1)) if m else -1
    cand.sort(key=itnum)
    ply_path = cand[-1]
    iter_dir = os.path.dirname(ply_path)
    return ply_path, iter_dir, itnum(ply_path)

def read_ply(path: str):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names

    def get(name):
        if name not in names:
            raise KeyError(f"Missing field {name} in PLY")
        return np.asarray(v[name], dtype=np.float32)

    pos = np.stack([get("x"), get("y"), get("z")], axis=1)
    if all(n in names for n in ["nx", "ny", "nz"]):
        nrm = np.stack([get("nx"), get("ny"), get("nz")], axis=1)
    else:
        nrm = np.zeros_like(pos)

    dc = np.stack([get("f_dc_0"), get("f_dc_1"), get("f_dc_2")], axis=1)

    rest_names = [n for n in names if n.startswith("f_rest_")]
    rest_names.sort(key=lambda s: int(s.split("_")[-1]))
    if len(rest_names) != 9:
        raise KeyError(f"Expect L1 => 9 f_rest_* (0..8), but got {len(rest_names)}: {rest_names[:12]}")
    rest = np.stack([get(n) for n in rest_names], axis=1)  # [N,9]

    opacity = get("opacity")[:, None]
    scale = np.stack([get("scale_0"), get("scale_1"), get("scale_2")], axis=1)
    rot = np.stack([get("rot_0"), get("rot_1"), get("rot_2"), get("rot_3")], axis=1)
    return pos, nrm, dc, rest, opacity, scale, rot

def quant_minmax(x: np.ndarray, bits: int = 8):
    qmax = (1 << bits) - 1
    x = x.astype(np.float32)
    mn = x.min(axis=0, keepdims=True)
    mx = x.max(axis=0, keepdims=True)
    span = (mx - mn)
    span[span < 1e-12] = 1.0
    q = np.round((x - mn) / span * qmax)
    q = q.astype(np.uint16 if bits > 8 else np.uint8)
    return q, mn.astype(np.float32), mx.astype(np.float32), np.int32(bits)

def quant_quat(rot: np.ndarray, bits: int = 16):
    rot = rot.astype(np.float32)
    n = np.linalg.norm(rot, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    rot = rot / n
    if bits == 16:
        q = np.round(np.clip(rot, -1, 1) * 32767.0).astype(np.int16)
    elif bits == 8:
        q = np.round(np.clip(rot, -1, 1) * 127.0).astype(np.int8)
    else:
        raise ValueError("rot_bits must be 8 or 16")
    return q, np.int32(bits)

# morton sort (better deflate)
def _part1by2(n):
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8))  & 0x0300F00F
    n = (n | (n << 4))  & 0x030C30C3
    n = (n | (n << 2))  & 0x09249249
    return n

def morton3D(xi, yi, zi):
    return (_part1by2(xi) | (_part1by2(yi) << 1) | (_part1by2(zi) << 2)).astype(np.uint32)

def morton_sort(pos: np.ndarray, bits: int = 10):
    mn = pos.min(axis=0)
    mx = pos.max(axis=0)
    span = np.maximum(mx - mn, 1e-12)
    grid = np.floor((pos - mn) / span * ((1 << bits) - 1)).astype(np.int32)
    grid = np.clip(grid, 0, (1 << bits) - 1).astype(np.uint32)
    code = morton3D(grid[:,0], grid[:,1], grid[:,2])
    return np.argsort(code, kind="stable")

@torch.no_grad()
def kmeans_torch(x: torch.Tensor, K: int, iters: int, seed: int):
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    N, D = x.shape
    idx = torch.randint(0, N, (K,), generator=g, device=x.device)
    c = x[idx].clone()
    bs = 200000
    for _ in range(iters):
        x2 = (x * x).sum(dim=1, keepdim=True)
        c2 = (c * c).sum(dim=1).view(1, K)
        labels = []
        for s in range(0, N, bs):
            xb = x[s:s+bs]
            d2 = x2[s:s+bs] + c2 - 2.0 * xb @ c.t()
            labels.append(torch.argmin(d2, dim=1))
        labels = torch.cat(labels, dim=0)
        c.zero_()
        counts = torch.zeros((K,), device=x.device, dtype=torch.float32)
        c.index_add_(0, labels, x)
        ones = torch.ones((N,), device=x.device, dtype=torch.float32)
        counts.index_add_(0, labels, ones)
        counts = torch.clamp(counts, min=1.0)
        c = c / counts[:, None]
    return c

def build_codebook(vec: np.ndarray, K: int, sample: int, iters: int, device: str, seed: int):
    N, D = vec.shape
    if sample > 0 and N > sample:
        rs = np.random.RandomState(seed)
        sel = rs.choice(N, size=sample, replace=False)
        train = vec[sel]
    else:
        train = vec
    xt = torch.from_numpy(train).to(device=device, dtype=torch.float32)
    cent = kmeans_torch(xt, K=K, iters=iters, seed=seed)  # [K,4]
    xfull = torch.from_numpy(vec).to(device=device, dtype=torch.float32)
    x2 = (xfull * xfull).sum(dim=1, keepdim=True)
    c2 = (cent * cent).sum(dim=1).view(1, K)
    bs = 200000
    idxs = []
    for s in range(0, N, bs):
        xb = xfull[s:s+bs]
        d2 = x2[s:s+bs] + c2 - 2.0 * xb @ cent.t()
        idxs.append(torch.argmin(d2, dim=1).cpu().numpy())
    idxs = np.concatenate(idxs, axis=0).astype(np.uint16 if K > 256 else np.uint8)
    return cent.cpu().numpy().astype(np.float16), idxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", default="cb256")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--kmeans_iters", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rot_bits", type=int, default=16, choices=[8,16])
    ap.add_argument("--no_morton", action="store_true")
    args = ap.parse_args()

    ply_path, iter_dir, itnum = find_latest_iter_ply(args.model_path)
    print(f"[ENC] latest iter = {itnum}")
    print(f"[ENC] read ply    = {ply_path}")

    pos, nrm, dc, rest, opacity, scale, rot = read_ply(ply_path)
    N = pos.shape[0]
    print(f"[ENC] gaussians   = {N}")

    # L1 layout: rest=[R(3),G(3),B(3)], per-channel vec=[dc, l1(3)] => 4D
    r = np.concatenate([dc[:,0:1], rest[:,0:3]], axis=1)
    g = np.concatenate([dc[:,1:2], rest[:,3:6]], axis=1)
    b = np.concatenate([dc[:,2:3], rest[:,6:9]], axis=1)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but --device=cuda was set.")

    print(f"[ENC] train codebook K={args.K} sample={args.sample} iters={args.kmeans_iters} dev={args.device}")
    cb_r, idx_r = build_codebook(r, args.K, args.sample, args.kmeans_iters, args.device, seed=0)
    cb_g, idx_g = build_codebook(g, args.K, args.sample, args.kmeans_iters, args.device, seed=1)
    cb_b, idx_b = build_codebook(b, args.K, args.sample, args.kmeans_iters, args.device, seed=2)

    pos16 = pos.astype(np.float16)
    nrm16 = nrm.astype(np.float16)
    op_q, op_mn, op_mx, op_bits = quant_minmax(opacity, bits=8)
    sc_q, sc_mn, sc_mx, sc_bits = quant_minmax(scale, bits=8)
    rot_q, rot_bits = quant_quat(rot, bits=args.rot_bits)

    perm = np.arange(N, dtype=np.int64) if args.no_morton else morton_sort(pos.astype(np.float32), bits=10)
    def ap(x): return x[perm]

    out_npz = os.path.join(iter_dir, f"point_cloud.{args.tag}.npz")
    np.savez_compressed(
        out_npz,
        degree=np.int32(1),
        K=np.int32(args.K),
        morton_used=np.int32(0 if args.no_morton else 1),
        cb_r=cb_r, cb_g=cb_g, cb_b=cb_b,
        idx_r=ap(idx_r), idx_g=ap(idx_g), idx_b=ap(idx_b),
        pos16=ap(pos16),
        nrm16=ap(nrm16),
        op_q=ap(op_q), op_mn=op_mn, op_mx=op_mx, op_bits=op_bits,
        sc_q=ap(sc_q), sc_mn=sc_mn, sc_mx=sc_mx, sc_bits=sc_bits,
        rot_q=ap(rot_q), rot_bits=rot_bits,
    )

    orig = os.path.getsize(ply_path)
    comp = os.path.getsize(out_npz)
    print(f"[ENC] saved npz   = {out_npz}")
    print(f"[ENC] ply size    = {orig/1024/1024:.2f} MB")
    print(f"[ENC] npz size    = {comp/1024/1024:.2f} MB  (ratio {orig/comp:.2f}x)")

if __name__ == "__main__":
    main()
