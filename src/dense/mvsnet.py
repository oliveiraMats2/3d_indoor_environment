# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

# PROMPT COMPLEMENT
# Stage: dense (method mvsnet) - MVSNet (Yao et al. ECCV 2018) learned MVS on the GPU.
# Goal: per reference view, infer a 1/4-resolution depth map with the official MVSNet weights (YoYo000/MVSNet,
#      TensorFlow, "3DCNNs" regularisation, BlendedMVS or DTU training) re-implemented in PyTorch, keep the
#      confident pixels and fuse with COLMAP like every other depth-map densifier.
# In:  <run_dir>/dense/undistorted (shared PINHOLE workspace), its sparse model (poses, depth range, sources);
#      the official tf_model_<weights>.zip (Google Drive link of the YoYo000/MVSNet README).
# Out: <run_dir>/dense/mvsnet/{ws/stereo/depth_maps, fused.ply, points.npy, colors.npy, colmap_dense.log};
#      row method=mvsnet in <artifacts_dir>/dense_metrics.csv (t_depth_s, vram_peak_gb);
#      third_party/weights/{tf_model_<weights>.zip, mvsnet_<weights>.pt} (converted once, gitignored).
# YAML: dense.mvsnet.{weights, max_size, num_views, num_depth, prob_threshold, depth_chunk};
#      dense.{enabled, methods, rerun, ref_step, gpu_index, fusion.*}.
# Rules: no CUDA / torch / download or conversion failure -> WARNING "official MVSNet weights unavailable"
#      and return None; the TF checkpoint is read without tensorflow (tensor-bundle .index table + .data
#      shard); the graph follows the official one: UNetDS2GN features, variance cost volume, RegNetUS0 with
#      batch statistics (the official test graph runs with is_training=True), softmax(-cost), soft argmin,
#      4-plane probability map, BGR input normalised per image, TF "SAME" padding; no refinement net.

# >>> BEGIN IMPLEMENTATION >>>
"""MVSNet depth maps (official TF weights converted to PyTorch) fused with COLMAP stereo_fusion."""
import logging
import struct
import zipfile
from pathlib import Path

import numpy as np

from src.dense import depth_fusion as df

log = logging.getLogger(__name__)

METHOD = "mvsnet"
REPO = ("YoYo000", "MVSNet", "3ae2cb2b72c6df58ebcb321d7d243d4efd01fbc5")   # MIT, reference graph
GDRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
WEIGHTS = {"blendedmvs": ("1HacSpLl49xB77uBuI67ceZV9Rx_8wjPV", "tf_model_blendedmvs_200301.zip",
                          "54060fe75036faf56481131457a68307dc5b8ea3ff277769370d879168e898f8"),
           "dtu": ("1-1JyFT9ClqPO0kz0d_5I1_IHX05paS4h", "tf_model_dtu_200301.zip",
                   "f9065192fdea8b765145d8b1da10b20e1a509267285fc634060aa9527879c641")}
CKPT = "3DCNNs/model.ckpt-150000"
BASE = 32
DEFAULTS = {"weights": "blendedmvs", "max_size": 1600, "num_views": 5, "num_depth": 192,
            "prob_threshold": 0.8, "depth_chunk": 16}

# --- TF tensor-bundle (checkpoint V2) reader, no tensorflow needed -------------------------------------

def _varint(buf: bytes, i: int) -> tuple[int, int]:

    out = shift = 0
    while True:
        b = buf[i]
        out |= (b & 0x7F) << shift
        i += 1
        if b < 0x80:
            return out, i
        shift += 7

def _proto(buf: bytes) -> dict[int, list]:
    """Minimal protobuf decoder: {field: [values]} (varint -> int, length-delimited -> bytes, fixed -> int)."""

    out, i = {}, 0
    while i < len(buf):
        tag, i = _varint(buf, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            val, i = buf[i:i + n], i + n
        elif wire == 5:
            val, i = struct.unpack_from("<I", buf, i)[0], i + 4
        elif wire == 1:
            val, i = struct.unpack_from("<Q", buf, i)[0], i + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.setdefault(field, []).append(val)
    return out

def _block(data: bytes, offset: int, size: int) -> list[tuple[bytes, bytes]]:
    """Entries of one uncompressed leveldb-style table block (prefix-compressed keys, restart array)."""

    if data[offset + size] != 0:
        raise ValueError("compressed checkpoint index blocks are not supported")
    blk = data[offset:offset + size]
    n_restarts = struct.unpack_from("<I", blk, len(blk) - 4)[0]
    end, i, key, out = len(blk) - 4 - 4 * n_restarts, 0, b"", []
    while i < end:
        shared, i = _varint(blk, i)
        non_shared, i = _varint(blk, i)
        vlen, i = _varint(blk, i)
        key = key[:shared] + blk[i:i + non_shared]
        i += non_shared
        out.append((key, blk[i:i + vlen]))
        i += vlen
    return out

def read_tf_checkpoint(prefix) -> dict[str, np.ndarray]:
    """Every float32 tensor of a TF V2 checkpoint (<prefix>.index + <prefix>.data-*) as numpy arrays."""

    data = Path(f"{prefix}.index").read_bytes()
    if struct.unpack_from("<Q", data, len(data) - 8)[0] != 0xDB4775248B80FB57:
        raise ValueError(f"{prefix}.index is not a TF table")
    i = len(data) - 48
    _, i = _varint(data, i)
    _, i = _varint(data, i)               # metaindex handle (unused)
    idx_off, i = _varint(data, i)
    idx_size, i = _varint(data, i)
    entries = []
    for _, handle in _block(data, idx_off, idx_size):
        off, j = _varint(handle, 0)
        size, _ = _varint(handle, j)
        entries += _block(data, off, size)
    header = _proto(dict(entries)[b""])
    n_shards = header.get(1, [1])[0]
    shards, out = {}, {}
    for key, val in entries:
        if not key:
            continue
        e = _proto(val)
        if e.get(1, [0])[0] != 1 or 7 in e:   # DT_FLOAT only, no partitioned variables
            continue
        shape = [_proto(d).get(1, [0])[0] for d in _proto(e[2][0]).get(2, [])] if 2 in e else []
        shard = e.get(3, [0])[0]
        if shard not in shards:
            shards[shard] = Path(f"{prefix}.data-{shard:05d}-of-{n_shards:05d}").read_bytes()
        off, size = e.get(4, [0])[0], e.get(5, [0])[0]
        out[key.decode()] = np.frombuffer(shards[shard][off:off + size], "<f4").reshape(shape).copy()
    return out

def convert(tf_vars: dict[str, np.ndarray]) -> dict:
    """Official TF variables -> {tf name: torch tensor}: conv/deconv kernels [k.., a, b] -> [b, a, k..]
    (TF conv [.., in, out] and conv_transpose [.., out, in] both map to the PyTorch layouts); optimizer
    slots (RMSProp), BN moving statistics (unused: batch statistics) and the global step are dropped."""

    import torch

    out, keep = {}, set(param_names())
    for name, arr in tf_vars.items():
        if name not in keep:
            continue
        if name.endswith("/kernel"):
            nd = arr.ndim - 2
            arr = arr.transpose(nd + 1, nd, *range(nd))
        out[name] = torch.from_numpy(np.ascontiguousarray(arr))
    return out

def weights_path(variant: str) -> Path:
    """Converted state dict of the official checkpoint (download + extract + convert once)."""

    import torch

    from src.dense import third_party

    if variant not in WEIGHTS:
        raise ValueError(f"dense.mvsnet.weights must be one of {sorted(WEIGHTS)}")
    out = third_party.ROOT / "weights" / f"mvsnet_{variant}.pt"
    if out.exists():
        return out
    gid, zip_name, sha = WEIGHTS[variant]
    zpath = third_party.fetch_file(GDRIVE.format(gid), zip_name, sha)
    tf_dir = third_party.ROOT / "weights" / f"mvsnet_tf_{variant}"
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.namelist():
            if f"/{CKPT}." in m and ".meta" not in m and ".." not in m:
                tf_dir.mkdir(parents=True, exist_ok=True)
                (tf_dir / Path(m).name).write_bytes(zf.read(m))
    state = convert(read_tf_checkpoint(tf_dir / Path(CKPT).name))
    if len(state) != len(param_names()):
        raise RuntimeError(f"checkpoint has {len(state)} usable tensors, expected {len(param_names())}")
    torch.save(state, out)
    log.info("mvsnet: converted %s -> %s (%d tensors)", zip_name, out, len(state))
    return out

# --- the official graph in PyTorch ---------------------------------------------------------------------

# (name, kernel, stride, kind) per layer; kind: gn/bn = conv+norm+relu, dgn = deconv+GN (no relu),
# dbn = deconv+BN+relu, plain = conv without bias/norm/relu (as cnn_wrapper/mvsnet.py)
FEATURE = {f"2dconv{a}_{b}": (3, 2 if b == 0 else 1, "gn") for a in range(1, 5) for b in range(3)}
FEATURE.update({"2dconv0_1": (3, 1, "gn"), "2dconv0_2": (3, 1, "gn"),
                **{f"2dconv{a}_0": (3, 2, "dgn") for a in range(5, 9)},
                **{f"2dconv{a}_{b}": (3, 1, "gn") for a in range(5, 9) for b in (1, 2)},
                "conv9_0": (5, 2, "gn"), "conv9_1": (3, 1, "gn"), "conv9_2": (3, 1, "gn"),
                "conv10_0": (5, 2, "gn"), "conv10_1": (3, 1, "gn"), "conv10_2": (3, 1, "plain")})
REG = {"3dconv0_1": (3, 1, "bn"), "3dconv1_0": (3, 2, "bn"), "3dconv1_1": (3, 1, "bn"),
       "3dconv2_0": (3, 2, "bn"), "3dconv2_1": (3, 1, "bn"), "3dconv3_0": (3, 2, "bn"),
       "3dconv3_1": (3, 1, "bn"), "3dconv4_0": (3, 2, "dbn"), "3dconv5_0": (3, 2, "dbn"),
       "3dconv6_0": (3, 2, "dbn"), "3dconv6_2": (3, 1, "plain")}

def param_names() -> list[str]:

    names = []
    for name, (_, _, kind) in {**FEATURE, **REG}.items():
        names.append(f"{name}/kernel")
        if kind != "plain":
            norm = "gn" if kind in ("gn", "dgn") else "bn"
            names += [f"{name}/{norm}/gamma", f"{name}/{norm}/beta"]
    return names

def _same_pad(n: int, k: int, s: int) -> tuple[int, int]:
    """TF 'SAME' padding (before, after) of one axis: out = ceil(n / s), the extra pixel goes after."""

    total = max((-(-n // s) - 1) * s + k - n, 0)
    return total // 2, total - total // 2

class MVSNet:
    """Functional MVSNet with weights keyed by the official TF variable names."""

    def __init__(self, state: dict, device):
        self.w = {k: v.to(device) for k, v in state.items()}

    def layer(self, x, name: str, table: dict):
        import torch.nn.functional as F

        k, s, kind = table[name]
        w = self.w[f"{name}/kernel"]
        dims = x.dim() - 2
        if kind in ("dgn", "dbn"):
            fn = F.conv_transpose2d if dims == 2 else F.conv_transpose3d
            y = fn(x, w, stride=s)
            # crop to TF conv_transpose 'SAME': out = n * s, starting at the forward conv's pad-before
            for ax in range(dims):
                n = x.shape[2 + ax]
                y = y.narrow(2 + ax, _same_pad(n * s, k, s)[0], n * s)
        else:
            pads = []
            for ax in reversed(range(dims)):
                pads += list(_same_pad(x.shape[2 + ax], k, s))
            fn = F.conv2d if dims == 2 else F.conv3d
            y = fn(F.pad(x, pads), w, stride=s)
        if kind == "plain":
            return y
        if kind in ("gn", "dgn"):
            y = F.group_norm(y, max(1, y.shape[1] // 8), self.w[f"{name}/gn/gamma"], self.w[f"{name}/gn/beta"],
                             eps=1e-5)
            return F.relu(y) if kind == "gn" else y
        y = F.batch_norm(y, None, None, self.w[f"{name}/bn/gamma"], self.w[f"{name}/bn/beta"], training=True,
                         eps=1e-5)
        return F.relu(y)

    def features(self, img):
        """UNetDS2GN: (B, 3, H, W) normalised BGR -> (B, 32, H/4, W/4)."""

        import torch

        f = lambda x, n: self.layer(x, n, FEATURE)
        c1 = f(img, "2dconv1_0")
        c2 = f(c1, "2dconv2_0")
        c3 = f(c2, "2dconv3_0")
        c4 = f(c3, "2dconv4_0")
        skip = [f(f(img, "2dconv0_1"), "2dconv0_2"), f(f(c1, "2dconv1_1"), "2dconv1_2"),
                f(f(c2, "2dconv2_1"), "2dconv2_2"), f(f(c3, "2dconv3_1"), "2dconv3_2")]
        x = f(f(f(c4, "2dconv4_1"), "2dconv4_2"), "2dconv5_0")
        for a, s in zip(range(5, 9), reversed(skip)):
            x = f(f(torch.cat([x, s], 1), f"2dconv{a}_1"), f"2dconv{a}_2")
            if a < 8:
                x = f(x, f"2dconv{a + 1}_0")
        for n in ("conv9_0", "conv9_1", "conv9_2", "conv10_0", "conv10_1", "conv10_2"):
            x = f(x, n)
        return x

    def regularise(self, cost):
        """RegNetUS0: (B, 32, D, H, W) variance volume -> (B, D, H, W) filtered cost."""

        f = lambda x, n: self.layer(x, n, REG)
        c1 = f(cost, "3dconv1_0")
        c2 = f(c1, "3dconv2_0")
        c3 = f(c2, "3dconv3_0")
        x = f(f(c3, "3dconv3_1"), "3dconv4_0") + f(c2, "3dconv2_1")
        x = f(x, "3dconv5_0") + f(c1, "3dconv1_1")
        x = f(x, "3dconv6_0") + f(cost, "3dconv0_1")
        return f(x, "3dconv6_2")[:, 0]

def center_image(bgr: np.ndarray) -> np.ndarray:
    """Official preprocess.center_image: per-channel zero mean / unit std over the image (0-255 input)."""

    img = bgr.astype(np.float32)
    return (img - img.mean((0, 1), keepdims=True)) / (img.std((0, 1), keepdims=True) + 1e-8)

def warp_grid(K_ref, Rt_ref, K_src, Rt_src, depths, h: int, w: int):
    """grid_sample grids (D, h, w, 2) of the source view for every fronto-parallel reference plane (same
    homography as get_homographies + tf_transform_homography: pixel centres at +0.5)."""

    import torch

    P1 = K_src @ Rt_src[:3, :3] @ Rt_ref[:3, :3].T @ torch.linalg.inv(K_ref)
    P2 = K_src @ (Rt_src[:3, 3] - Rt_src[:3, :3] @ Rt_ref[:3, :3].T @ Rt_ref[:3, 3])
    v, u = torch.meshgrid(torch.arange(h, device=depths.device, dtype=torch.float64) + 0.5,
                          torch.arange(w, device=depths.device, dtype=torch.float64) + 0.5, indexing="ij")
    p = torch.stack([u, v, torch.ones_like(u)], -1).reshape(-1, 3) @ P1.T            # (hw, 3)
    q = depths[:, None, None] * p[None] + P2[None, None]                               # (D, hw, 3)
    z = q[..., 2]
    ok = z > 1e-6
    z = torch.where(ok, z, torch.ones_like(z))
    x = torch.where(ok, q[..., 0] / z, torch.full_like(z, -1e6))
    y = torch.where(ok, q[..., 1] / z, torch.full_like(z, -1e6))
    grid = torch.stack([x / w * 2 - 1, y / h * 2 - 1], -1).clamp(-2, 2)
    return grid.reshape(len(depths), h, w, 2).float()

def infer(net: MVSNet, imgs: np.ndarray, Ks: list, Rts: list, depths: np.ndarray, chunk: int, device):
    """One reference: imgs (N, 3, H, W) centred BGR (ref first), K at input resolution, depth planes ->
    (depth, probability) at (H/4, W/4)."""

    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        feats = net.features(torch.from_numpy(imgs).to(device))
        n, c, h, w = feats.shape
        d = torch.from_numpy(depths).to(device, torch.float64)
        K4 = [torch.from_numpy(df.scale_K(K, 0.25, 0.25)).to(device) for K in Ks]
        Rt = [torch.from_numpy(np.asarray(m, np.float64)).to(device) for m in Rts]
        cost = torch.empty((1, c, len(depths), h, w), device=device)
        for a in range(0, len(depths), chunk):
            dc = d[a:a + chunk]
            s1 = feats[0:1, :, None].expand(-1, -1, len(dc), -1, -1).clone()
            s2 = s1 ** 2
            for i in range(1, n):
                g = warp_grid(K4[0], Rt[0], K4[i], Rt[i], dc, h, w)
                warped = F.grid_sample(feats[i:i + 1].expand(len(dc), -1, -1, -1), g, mode="bilinear",
                                       padding_mode="zeros", align_corners=False)
                warped = warped.permute(1, 0, 2, 3)[None]
                s1 += warped
                s2 += warped ** 2
                del warped
            cost[:, :, a:a + chunk] = s2 / n - (s1 / n) ** 2
            del s1, s2
        del feats
        prob = torch.softmax(-net.regularise(cost), 1)[0]                                  # (D, h, w)
        del cost
        df_ = d.float()
        depth = torch.sum(prob * df_[:, None, None], 0)
        # official get_propability_map: sum over the planes floor-1, floor, ceil, ceil+1
        idx = (depth - df_[0]) / (df_[1] - df_[0])
        lo, hi = idx.floor().long(), idx.ceil().long()
        conf = sum(prob.gather(0, j.clamp(0, len(depths) - 1)[None])[0] for j in (lo - 1, lo, hi, hi + 1))
    return depth.cpu().numpy(), conf.cpu().numpy()

def input_size(width: int, height: int, max_size: int, base: int = BASE) -> tuple[int, int]:
    """Network input (w, h): long side <= max_size, both floored to a multiple of 32 (the 2D U-Net downsamples
    by 16 and the 3D U-Net the 1/4 feature map by 8)."""

    s = min(1.0, max_size / max(width, height))
    return max(base, int(width * s) // base * base), max(base, int(height * s) // base * base)

def prepare(ws, views: dict, ref: str, p: dict, cache: dict) -> tuple | None:
    """Network inputs of one reference (None when it has no sparse depth range or no source view)."""

    import cv2

    v = views[ref]
    srcs = [s for s in v.sources if s in views][:int(p["num_views"]) - 1]
    if v.depth_max <= v.depth_min or v.depth_min <= 0 or not srcs:
        return None
    w, h = input_size(v.width, v.height, int(p["max_size"]))
    imgs, Ks, Rts = [], [], []
    for name in [ref] + srcs:
        u = views[name]
        if (name, w, h) not in cache:
            cache.clear() if len(cache) > 64 else None
            rgb = cv2.resize(df.read_image(ws, name), (w, h), interpolation=cv2.INTER_AREA)
            cache[(name, w, h)] = center_image(rgb[:, :, ::-1] * 255.0).transpose(2, 0, 1).copy()
        imgs.append(cache[(name, w, h)])
        Ks.append(df.scale_K(u.K, w / u.width, h / u.height))
        Rt = np.eye(4)
        Rt[:3, :3], Rt[:3, 3] = u.R, u.t
        Rts.append(Rt)
    n = int(p["num_depth"]) // 8 * 8
    depths = np.linspace(v.depth_min, v.depth_max, n)
    return np.stack(imgs).astype(np.float32), Ks, Rts, depths

def load_net(p: dict, device) -> MVSNet:

    import torch

    state = torch.load(weights_path(str(p["weights"])), map_location="cpu", weights_only=True)
    missing = set(param_names()) - set(state)
    if missing:
        raise RuntimeError(f"converted weights miss {sorted(missing)[:3]}")
    return MVSNet(state, device)

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> Path | None:

    if not df.enabled(cfg, METHOD):
        return None
    p = df.method_cfg(cfg, METHOD, DEFAULTS)
    row = df.base_row(cfg, METHOD)
    row["backend"] = f"torch ({p['weights']})"
    if (fused := df.reuse(cfg, run_dir, artifacts_dir, METHOD, row)) is not None:
        return fused
    try:
        import torch
    except ImportError:
        log.warning("dense mvsnet skipped: torch not installed")
        return None
    if not torch.cuda.is_available():
        log.warning("dense mvsnet skipped: CUDA not available")
        return None
    undist = df.undistorted_workspace(cfg, run_dir)
    if undist is None:
        return None
    device = torch.device(f"cuda:{str(cfg['dense'].get('gpu_index', 0)).split(',')[0]}")
    try:
        net = load_net(p, device)
    except Exception as exc:  # download, unzip, checkpoint format or name mismatch
        log.warning("official MVSNet weights unavailable: %s", exc)
        return None

    views = df.load_views(undist, max(df.N_SOURCES, int(p["num_views"]) - 1))
    refs = [r for r in df.references(cfg, undist) if r in views]
    ws = df.method_workspace(run_dir, METHOD, undist)
    for old in (ws / "stereo" / "depth_maps").glob("*.bin"):
        old.unlink()
    torch.cuda.reset_peak_memory_stats(device)
    cache, kept = {}, []
    with df.Timer() as t:
        for i, ref in enumerate(refs):
            inputs = prepare(undist, views, ref, p, cache)
            if inputs is None:
                log.warning("dense mvsnet: %s has no depth range or sources, skipped", ref)
                continue
            imgs, Ks, Rts, depths = inputs
            try:
                depth, prob = infer(net, imgs, Ks, Rts, depths, int(p["depth_chunk"]), device)
            except torch.cuda.OutOfMemoryError:
                log.warning("dense mvsnet: out of GPU memory at %s (lower dense.mvsnet.max_size or num_depth)", ref)
                torch.cuda.empty_cache()
                break
            depth[(prob < float(p["prob_threshold"])) | ~np.isfinite(depth)] = 0
            depth[(depth < depths[0]) | (depth > depths[-1])] = 0
            df.save_depth(ws, ref, depth, df.scale_K(Ks[0], 0.25, 0.25))
            kept.append(float((depth > 0).mean()))
            if i % 10 == 0:
                log.info("mvsnet %d/%d %s: %.0f%% confident", i + 1, len(refs), ref, 100 * kept[-1])
    row["t_depth_s"] = round(t.s, 2)
    row["vram_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    row["n_excluded"] = len(views) - len(refs)
    log.info("mvsnet: %d depth maps in %.1f s, mean confident fraction %.2f", len(kept), t.s,
             float(np.mean(kept)) if kept else 0.0)
    del net
    torch.cuda.empty_cache()
    return df.fuse_and_export(cfg, run_dir, artifacts_dir, METHOD, ws, refs, row)

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    assert input_size(1600, 900, 1600) == (1600, 896)
    assert _same_pad(8, 3, 2) == (0, 1) and _same_pad(8, 5, 2) == (1, 2) and _same_pad(7, 3, 1) == (1, 1)
    assert len(param_names()) == 3 * 31 + 1 + 3 * 10 + 1
    # protobuf / varint round trip of a BundleEntryProto-like message
    msg = bytes([0x08, 0x01, 0x20, 0x96, 0x01, 0x28, 0x10])
    assert _proto(msg) == {1: [1], 4: [150], 5: [16]}

    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        net = load_net(DEFAULTS, dev)
        # fronto-parallel textured plane at z=2 seen by 3 translated cameras -> depth close to 2
        h, w, n = 128, 160, 3
        K = np.array([[150.0, 0, w / 2], [0, 150, h / 2], [0, 0, 1]])
        tex = np.random.default_rng(0).random((h, 3 * w, 3)).astype(np.float32)
        tex = np.kron(tex[::4, ::4], np.ones((4, 4, 1), np.float32))
        Rts, imgs = [], []
        for k in range(n):
            Rt = np.eye(4)
            Rt[0, 3] = -0.1 * k
            Rts.append(Rt)
            shift = int(round(150 * 0.1 * k / 2.0))          # disparity of the plane, pixels
            imgs.append(center_image(tex[:, w + shift:2 * w + shift] * 255).transpose(2, 0, 1))
        depth, prob = infer(net, np.stack(imgs), [K] * n, Rts, np.linspace(1.0, 4.0, 64), 16, dev)
        assert depth.shape == (h // 4, w // 4) and np.all(np.isfinite(depth))
        med = float(np.median(depth[4:-4, 4:-12]))
        print("plane depth median", med, "prob mean", float(prob.mean()))
        assert abs(med - 2.0) < 0.15, med
        assert 0 <= prob.min() and prob.max() <= 1.0 + 1e-4
    print("mvsnet self-test OK")
