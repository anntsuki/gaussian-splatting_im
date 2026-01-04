#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import torch
from plyfile import PlyData

# ----------- 复用你的 PLY 读取逻辑（精简版，和 gs_encode.py 一致） -----------
def read_ply_sh(path: str):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names

    def get(name):
        if name not in names:
            raise KeyError(f"Missing field {name} in PLY")
        return np.asarray(v[name], dtype=np.float32)

    dc = np.stack([get("f_dc_0"), get("f_dc_1"), get("f_dc_2")], axis=1)

    rest_names = [n for n in names if n.startswith("f_rest_")]
    rest_names.sort(key=lambda s: int(s.split("_")[-1]))
    rest = np.stack([get(n) for n in rest_names], axis=1)  # [N, 3*C] (interleaved)

    N = dc.shape[0]
    n_rest = rest.shape[1]
    if n_rest % 3 != 0:
        raise ValueError(f"Rest coeffs {n_rest} not divisible by 3")
    n_coeffs = n_rest // 3

    # interleaved -> planar per channel
    rest_reshaped = rest.reshape(N, n_coeffs, 3)
    rest_r = rest_reshaped[:, :, 0]
    rest_g = rest_reshaped[:, :, 1]
    rest_b = rest_reshaped[:, :, 2]

    # concat DC + Rest per channel: shape [N, 1+C]
    r = np.concatenate([dc[:, 0:1], rest_r], axis=1)
    g = np.concatenate([dc[:, 1:2], rest_g], axis=1)
    b = np.concatenate([dc[:, 2:3], rest_b], axis=1)
    return r, g, b  # float32


def nearest_idx_torch(x: torch.Tensor, cb: torch.Tensor, bs: int = 10000):
    """x: [N,D], cb: [K,D] -> idx: [N]"""
    N, D = x.shape
    K = cb.shape[0]
    x2 = (x * x).sum(dim=1, keepdim=True)      # [N,1]
    c2 = (cb * cb).sum(dim=1).view(1, K)       # [1,K]
    idxs = []
    for s in range(0, N, bs):
        xb = x[s:s+bs]
        d2 = x2[s:s+bs] + c2 - 2.0 * xb @ cb.t()
        idxs.append(torch.argmin(d2, dim=1))
    return torch.cat(idxs, dim=0)


def finetune_codebook(
    x: torch.Tensor,         # [N,D] float32
    cb_init: torch.Tensor,   # [K,D] float32
    idx_init: torch.Tensor,  # [N] long
    steps: int,
    lr: float,
    batch: int,
    reassign_every: int = 0,
    device: str = "cuda",
):
    x = x.to(device)
    cb = torch.nn.Parameter(cb_init.to(device))
    idx = idx_init.to(device)

    opt = torch.optim.Adam([cb], lr=lr)

    N = x.shape[0]
    for step in range(1, steps + 1):
        # optional reassign
        if reassign_every and (step == 1 or step % reassign_every == 0):
            with torch.no_grad():
                idx = nearest_idx_torch(x, cb)

        # minibatch
        sel = torch.randint(0, N, (min(batch, N),), device=device)
        xb = x[sel]                 # [B,D]
        ib = idx[sel]               # [B]
        recon = cb[ib]              # [B,D]

        loss = torch.mean((recon - xb) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1 or step == steps:
            print(f"[CB-FT] step {step:5d}/{steps} loss={loss.item():.6f}")

    return cb.detach().cpu(), idx.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="original point_cloud.ply (uncompressed)")
    ap.add_argument("--npz", required=True, help="encoded npz from gs_encode.py")
    ap.add_argument("--out", required=True, help="output npz path (e.g. point_cloud.cb256_ft.npz)")

    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--reassign_every", type=int, default=0, help="0=fix idx; else reassign periodically")

    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but --device=cuda was set.")

    # load original vectors
    r, g, b = read_ply_sh(args.ply)
    r = torch.from_numpy(r.astype(np.float32))
    g = torch.from_numpy(g.astype(np.float32))
    b = torch.from_numpy(b.astype(np.float32))

    z = np.load(args.npz, allow_pickle=False)

    cb_r = torch.from_numpy(z["cb_r"].astype(np.float32))
    cb_g = torch.from_numpy(z["cb_g"].astype(np.float32))
    cb_b = torch.from_numpy(z["cb_b"].astype(np.float32))

    idx_r = torch.from_numpy(z["idx_r"].astype(np.int64))
    idx_g = torch.from_numpy(z["idx_g"].astype(np.int64))
    idx_b = torch.from_numpy(z["idx_b"].astype(np.int64))

    print("[CB-FT] finetune R codebook...")
    cb_r2, idx_r2 = finetune_codebook(r, cb_r, idx_r, args.steps, args.lr, args.batch, args.reassign_every, args.device)
    print("[CB-FT] finetune G codebook...")
    cb_g2, idx_g2 = finetune_codebook(g, cb_g, idx_g, args.steps, args.lr, args.batch, args.reassign_every, args.device)
    print("[CB-FT] finetune B codebook...")
    cb_b2, idx_b2 = finetune_codebook(b, cb_b, idx_b, args.steps, args.lr, args.batch, args.reassign_every, args.device)

    # save: keep everything else identical, only replace cb_*/idx_* if updated
    out_dict = {k: z[k] for k in z.files}
    out_dict["cb_r"] = cb_r2.numpy().astype(z["cb_r"].dtype)
    out_dict["cb_g"] = cb_g2.numpy().astype(z["cb_g"].dtype)
    out_dict["cb_b"] = cb_b2.numpy().astype(z["cb_b"].dtype)

    # 如果你开启 reassign，就也写回 idx；否则保持原 idx
    if args.reassign_every:
        out_dict["idx_r"] = idx_r2.numpy().astype(z["idx_r"].dtype)
        out_dict["idx_g"] = idx_g2.numpy().astype(z["idx_g"].dtype)
        out_dict["idx_b"] = idx_b2.numpy().astype(z["idx_b"].dtype)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out_dict)
    print(f"[CB-FT] saved: {args.out}")


if __name__ == "__main__":
    main()
