# decompress_gs.py
import os, argparse
import numpy as np
from plyfile import PlyData, PlyElement

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
    arr["opacity"] = opacity[:, 0]
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
    ap.add_argument("--inp", type=str, required=True)
    ap.add_argument("--out_ply", type=str, required=True)
    args = ap.parse_args()

    z = np.load(args.inp, allow_pickle=False)
    K = int(z["K"])
    cb_r = z["cb_r"].astype(np.float32)  # [K,4]
    cb_g = z["cb_g"].astype(np.float32)
    cb_b = z["cb_b"].astype(np.float32)
    idx_r = z["idx_r"].astype(np.int64)
    idx_g = z["idx_g"].astype(np.int64)
    idx_b = z["idx_b"].astype(np.int64)

    # reconstruct L1 per-channel [dc, l1(3)]
    r = cb_r[idx_r]  # [N,4]
    g = cb_g[idx_g]
    b = cb_b[idx_b]

    dc = np.stack([r[:,0], g[:,0], b[:,0]], axis=1).astype(np.float32)  # [N,3]
    rest = np.concatenate([r[:,1:4], g[:,1:4], b[:,1:4]], axis=1).astype(np.float32)  # [N,9]

    pos = z["pos16"].astype(np.float32)
    nrm = z["nrm16"].astype(np.float32) if "nrm16" in z else np.zeros_like(pos)

    op_bits = int(z["op_bits"]); sc_bits = int(z["sc_bits"])
    opacity = dequant_minmax(z["op_q"], z["op_mn"], z["op_mx"], op_bits).astype(np.float32)
    scale = dequant_minmax(z["sc_q"], z["sc_mn"], z["sc_mx"], sc_bits).astype(np.float32)

    rot_bits = int(z["rot_bits"])
    rot = dequant_quat(z["rot_q"], rot_bits).astype(np.float32)

    os.makedirs(os.path.dirname(args.out_ply) or ".", exist_ok=True)
    write_ply(args.out_ply, pos, nrm, dc, rest, opacity, scale, rot)
    print("Wrote:", args.out_ply)

if __name__ == "__main__":
    main()
