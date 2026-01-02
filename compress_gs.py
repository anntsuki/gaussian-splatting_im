# compress_gs.py
import os, re, glob, argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement

# ----------------- utils -----------------
def find_latest_ply(model_path: str) -> str:
    # 兼容：model_path/point_cloud/iteration_XXXX/point_cloud.ply
    cand = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply"))
    if not cand:
        # 有些人会直接放 point_cloud.ply
        cand2 = glob.glob(os.path.join(model_path, "*.ply"))
        if cand2:
            return cand2[0]
        raise FileNotFoundError(f"Cannot find point_cloud.ply under {model_path}")
    def itnum(p):
        m = re.search(r"iteration_(\d+)", p)
        return int(m.group(1)) if m else -1
    cand.sort(key=itnum)
    return cand[-1]

def read_ply(path: str):
    ply = PlyData.read(path)
    v = ply["vertex"].data  # structured numpy array
    names = v.dtype.names
    def get(name):
        if name not in names:
            raise KeyError(f"Missing field {name} in PLY")
        return np.asarray(v[name], dtype=np.float32)

    pos = np.stack([get("x"), get("y"), get("z")], axis=1)

    # 3DGS standard fields
    dc = np.stack([get("f_dc_0"), get("f_dc_1"), get("f_dc_2")], axis=1)

    # L1: rest = 9 dims => f_rest_0..8
    rest_names = [f"f_rest_{i}" for i in range(9)]
    if not all(n in names for n in rest_names):
        # 尝试自动推断 f_rest 数量
        rest_auto = [n for n in names if n.startswith("f_rest_")]
        rest_auto.sort(key=lambda s: int(s.split("_")[-1]))
        if len(rest_auto) != 9:
            raise KeyError(f"Expect L1 => 9 f_rest_*, but got {len(rest_auto)}. Fields: {rest_auto[:10]}")
        rest_names = rest_auto
    rest = np.stack([get(n) for n in rest_names], axis=1)  # [N, 9]

    opacity = get("opacity")[:, None]  # [N,1]
    scale = np.stack([get("scale_0"), get("scale_1"), get("scale_2")], axis=1)
    rot = np.stack([get("rot_0"), get("rot_1"), get("rot_2"), get("rot_3")], axis=1)

    # 一些 ply 会带 nx/ny/nz，保留（没有就补0）
    if all(n in names for n in ["nx", "ny", "nz"]):
        nrm = np.stack([get("nx"), get("ny"), get("nz")], axis=1)
    else:
        nrm = np.zeros_like(pos)

    return pos, nrm, dc, rest, opacity, scale, rot

def write_ply(path: str, pos, nrm, dc, rest, opacity, scale, rot):
    N = pos.shape[0]
    # 构造 structured array
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
    arr["x"], arr["y"], arr["z"] = pos[:,0], pos[:,1], pos[:,2]
    arr["nx"], arr["ny"], arr["nz"] = nrm[:,0], nrm[:,1], nrm[:,2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc[:,0], dc[:,1], dc[:,2]
    for i in range(rest.shape[1]):
        arr[f"f_rest_{i}"] = rest[:, i]
    arr["opacity"] = opacity[:, 0]
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scale[:,0], scale[:,1], scale[:,2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rot[:,0], rot[:,1], rot[:,2], rot[:,3]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(path)

def quant_minmax(x: np.ndarray, bits: int = 8):
    # per-dimension min-max
    qmax = (1 << bits) - 1
    x = x.astype(np.float32)
    mn = x.min(axis=0, keepdims=True)
    mx = x.max(axis=0, keepdims=True)
    scale = (mx - mn)
    scale[scale < 1e-12] = 1.0
    q = np.round((x - mn) / scale * qmax).astype(np.uint16 if bits > 8 else np.uint8)
    return q, mn.astype(np.float32), mx.astype(np.float32), bits

def dequant_minmax(q: np.ndarray, mn: np.ndarray, mx: np.ndarray, bits: int):
    qmax = (1 << bits) - 1
    q = q.astype(np.float32)
    return mn + (mx - mn) * (q / qmax)

def quant_quat(rot: np.ndarray, bits: int = 16):
    # rot in [-1,1], store signed int
    rot = rot.astype(np.float32)
    # normalize just in case
    n = np.linalg.norm(rot, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    rot = rot / n
    if bits == 16:
        q = np.round(np.clip(rot, -1, 1) * 32767.0).astype(np.int16)
    elif bits == 8:
        q = np.round(np.clip(rot, -1, 1) * 127.0).astype(np.int8)
    else:
        raise ValueError("rot_bits must be 8 or 16")
    return q, bits

def dequant_quat(q: np.ndarray, bits: int):
    if bits == 16:
        rot = q.astype(np.float32) / 32767.0
    else:
        rot = q.astype(np.float32) / 127.0
    n = np.linalg.norm(rot, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return rot / n

# Morton code for zlib friendliness (10 bits/axis -> 30-bit code)
def _part1by2(n):
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8))  & 0x0300F00F
    n = (n | (n << 4))  & 0x030C30C3
    n = (n | (n << 2))  & 0x09249249
    return n

def morton3D(xi, yi, zi):
    return (_part1by2(xi) | (_part1by2(yi) << 1) | (_part1by2(zi) << 2)).astype(np.uint32)

def morton_sort(pos: np.ndarray, bits: int = 10):
    # normalize to [0, 2^bits-1]
    mn = pos.min(axis=0)
    mx = pos.max(axis=0)
    span = np.maximum(mx - mn, 1e-12)
    grid = np.floor((pos - mn) / span * ((1 << bits) - 1)).astype(np.int32)
    grid = np.clip(grid, 0, (1 << bits) - 1).astype(np.uint32)
    code = morton3D(grid[:,0], grid[:,1], grid[:,2])
    perm = np.argsort(code, kind="stable")
    return perm

# ----------------- KMeans (torch) -----------------
@torch.no_grad()
def kmeans_torch(x: torch.Tensor, K: int, iters: int = 25, seed: int = 0):
    # x: [N, D] on GPU/CPU
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    N, D = x.shape
    # init: random pick
    idx = torch.randint(0, N, (K,), generator=g, device=x.device)
    c = x[idx].clone()  # [K,D]

    for _ in range(iters):
        # assign (chunk to save mem)
        # dist^2 = ||x||^2 + ||c||^2 - 2 x c^T
        x2 = (x * x).sum(dim=1, keepdim=True)  # [N,1]
        c2 = (c * c).sum(dim=1).view(1, K)     # [1,K]
        # compute in one shot (N*K might be large but D=4 so ok; still chunk safe)
        bs = 200000
        labels = []
        for s in range(0, N, bs):
            xb = x[s:s+bs]
            d2 = x2[s:s+bs] + c2 - 2.0 * xb @ c.t()
            labels.append(torch.argmin(d2, dim=1))
        labels = torch.cat(labels, dim=0)  # [N]

        # update
        c.zero_()
        counts = torch.zeros((K,), device=x.device, dtype=torch.float32)
        c.index_add_(0, labels, x)
        ones = torch.ones((N,), device=x.device, dtype=torch.float32)
        counts.index_add_(0, labels, ones)
        counts = torch.clamp(counts, min=1.0)
        c = c / counts[:, None]
    return c, labels

def build_codebook_per_channel(vec: np.ndarray, K: int, sample: int, iters: int, device: str, seed: int):
    N, D = vec.shape
    if sample > 0 and N > sample:
        rs = np.random.RandomState(seed)
        sel = rs.choice(N, size=sample, replace=False)
        train = vec[sel]
    else:
        train = vec

    xt = torch.from_numpy(train).to(device=device, dtype=torch.float32)
    centroids, _ = kmeans_torch(xt, K=K, iters=iters, seed=seed)

    # full assignment (chunk)
    xfull = torch.from_numpy(vec).to(device=device, dtype=torch.float32)
    c = centroids
    x2 = (xfull * xfull).sum(dim=1, keepdim=True)
    c2 = (c * c).sum(dim=1).view(1, K)
    bs = 200000
    idxs = []
    with torch.no_grad():
        for s in range(0, N, bs):
            xb = xfull[s:s+bs]
            d2 = x2[s:s+bs] + c2 - 2.0 * xb @ c.t()
            idxs.append(torch.argmin(d2, dim=1).cpu().numpy())
    idxs = np.concatenate(idxs, axis=0).astype(np.uint16 if K > 256 else np.uint8)
    return centroids.cpu().numpy().astype(np.float16), idxs

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--ply", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--sample", type=int, default=200000, help="sample points for kmeans training")
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--rot_bits", type=int, default=16, choices=[8,16])
    ap.add_argument("--no_morton", action="store_true")
    args = ap.parse_args()

    if args.ply is None:
        if args.model_path is None:
            raise ValueError("Provide --ply or --model_path")
        ply_path = find_latest_ply(args.model_path)
    else:
        ply_path = args.ply

    print("Reading:", ply_path)
    pos, nrm, dc, rest, opacity, scale, rot = read_ply(ply_path)
    N = pos.shape[0]
    print("Gaussians:", N)

    # L1 channel vectors: [dc, l1(3)]
    # rest layout: [R(3), G(3), B(3)]
    r = np.concatenate([dc[:,0:1], rest[:,0:3]], axis=1)  # [N,4]
    g = np.concatenate([dc[:,1:2], rest[:,3:6]], axis=1)
    b = np.concatenate([dc[:,2:3], rest[:,6:9]], axis=1)

    # codebooks
    print(f"Training codebooks K={args.K} sample={args.sample} iters={args.kmeans_iters} on {args.device}")
    cb_r, idx_r = build_codebook_per_channel(r, args.K, args.sample, args.kmeans_iters, args.device, seed=0)
    cb_g, idx_g = build_codebook_per_channel(g, args.K, args.sample, args.kmeans_iters, args.device, seed=1)
    cb_b, idx_b = build_codebook_per_channel(b, args.K, args.sample, args.kmeans_iters, args.device, seed=2)

    # quantize other params
    pos16 = pos.astype(np.float16)
    op_q, op_mn, op_mx, op_bits = quant_minmax(opacity, bits=8)
    sc_q, sc_mn, sc_mx, sc_bits = quant_minmax(scale, bits=8)
    rot_q, rot_bits = quant_quat(rot, bits=args.rot_bits)

    # morton sort for better deflate
    if args.no_morton:
        perm = np.arange(N, dtype=np.int64)
    else:
        perm = morton_sort(pos.astype(np.float32), bits=10)

    def apply_perm(x):
        return x[perm]

    payload = dict(
        # codebooks + indices
        K=np.int32(args.K),
        cb_r=cb_r, cb_g=cb_g, cb_b=cb_b,
        idx_r=apply_perm(idx_r), idx_g=apply_perm(idx_g), idx_b=apply_perm(idx_b),

        # geometry/other params
        pos16=apply_perm(pos16),
        nrm16=apply_perm(nrm.astype(np.float16)),  # optional
        op_q=apply_perm(op_q), op_mn=op_mn, op_mx=op_mx, op_bits=np.int32(op_bits),
        sc_q=apply_perm(sc_q), sc_mn=sc_mn, sc_mx=sc_mx, sc_bits=np.int32(sc_bits),
        rot_q=apply_perm(rot_q), rot_bits=np.int32(rot_bits),

        # metadata
        degree=np.int32(1),
        morton_used=np.int32(0 if args.no_morton else 1),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
