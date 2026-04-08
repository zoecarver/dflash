"""Shared device infrastructure for Tenstorrent 4-chip TP.

Provides: device open/close, tensor placement (rep/shd/rb), padding,
and shared building blocks (dev_norm, dev_add) used by both target and draft models.
"""
import torch
import torch.nn.functional as F
import ttnn
from rmsnorm import make_rmsnorm_kernel
from residual_add import residual_add_kernel
from silu_mul import silu_mul_kernel
from per_head_rmsnorm import make_per_head_rmsnorm_kernel
from rope import make_rope_kernel

TILE = 32
HIDDEN = 2048
HTILES = HIDDEN // TILE
HDIM = 128
HDIM_TILES = HDIM // TILE  # 4
NQH = 32
NKVH = 4
GQA = NQH // NKVH  # 8
EPS = 1e-6
ROPE_THETA = 1e7
VOCAB = 151936

N_CHIPS = 4
NQH_TP = NQH // N_CHIPS   # 8 Q heads per chip
NKVH_TP = NKVH // N_CHIPS  # 1 KV head per chip
Q_TP = NQH_TP * HDIM       # 1024
KV_TP = NKVH_TP * HDIM     # 128

TARGET_DIR = "/home/zcarver/qwen-coder-30b-a3b"

# Kernel instances
rmsnorm_k = make_rmsnorm_kernel(dim_tiles=HTILES, eps=EPS)
# Per-head RMSNorm kernels: PCC ~0.915 on mesh (vs >0.9999 in unit test on single device).
# Using ttnn.reshape + ttnn.rms_norm instead for now. Investigate bf16 reduction accuracy.
q_head_norm_k = make_per_head_rmsnorm_kernel(head_tiles=HDIM_TILES, n_heads=NQH_TP, eps=EPS)
k_head_norm_k = make_per_head_rmsnorm_kernel(head_tiles=HDIM_TILES, n_heads=NKVH_TP, eps=EPS)
q_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NQH_TP)
k_rope_k = make_rope_kernel(head_tiles=HDIM_TILES, n_heads=NKVH_TP)

_MESH = None


def open_dev():
    global _MESH
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    _MESH = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS))
    return _MESH


def close_dev(d):
    global _MESH
    ttnn.close_mesh_device(_MESH)
    _MESH = None


def _p(t):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    ph, pw = (TILE - h % TILE) % TILE, (TILE - w % TILE) % TILE
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph))
    return t.contiguous().to(torch.bfloat16)


def _mk(d):
    if isinstance(d, ttnn.MeshDevice):
        return {"mesh_mapper": ttnn.ReplicateTensorToMesh(d)}
    return {}


def rep(t, d, mem=None):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG, **_mk(d))


def shd(t, d, dim, mem=None):
    return ttnn.from_torch(_p(t), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(d, dim=dim))


def ztt(shape, d):
    return rep(torch.zeros(shape, dtype=torch.bfloat16), d)


def rb(t):
    if _MESH:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH, dim=0))[:t.shape[0]]
    return ttnn.to_torch(t)


def rb_dim1(t):
    """Readback column-sharded tensor, concatenating chips along dim=1."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(_MESH, dim=1))


# ---------------------------------------------------------------------------
# On-device building blocks (used by both target and draft)
# ---------------------------------------------------------------------------
def dev_norm(x, nw_key, w, out):
    """RMSNorm on device. nw_key is key into w for precomputed device weight."""
    rmsnorm_k(x, w[nw_key], w["sc"], w["ms"], out)
    return out


def dev_add(a, b, out):
    residual_add_kernel(a, b, out)
    return out
