"""Compute device selection, GPU memory based shot batching, and an environment check."""
from __future__ import annotations

import time

import numpy as np
import torch

# Deepwave elastic wavefields per shot: velocities, stresses and their PML memory variables
FIELDS_PER_SHOT = {2: 13, 3: 27}


def resolve_device(name: str | None) -> str:
    """'auto' -> 'cuda' when a CUDA GPU is visible, else 'cpu'. 'cuda', 'cuda:1', 'cpu' pass through."""
    name = (name or "auto").lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch sees no CUDA GPU (torch.cuda.is_available() is False). "
                           "Check the NVIDIA driver and that torch was installed with CUDA support.")
    return name


def is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def sync(device: str) -> None:
    """Wait for queued GPU work, so wall-clock timings are real."""
    if is_cuda(device):
        torch.cuda.synchronize(device)


def padded_cells(shape, meta: dict) -> int:
    """Cells Deepwave actually allocates: model + PML + finite-difference halo on every side."""
    pad = int(meta["grid"]["pml_width"]) + int(meta["grid"]["accuracy"]) // 2
    return int(np.prod([n + 2 * pad for n in shape]))


def bytes_per_shot(shape, meta: dict) -> int:
    dims = len(shape)
    safety = float(meta.get("sim", {}).get("memory_safety", 2.0))
    return int(padded_cells(shape, meta) * FIELDS_PER_SHOT[dims] * 4 * safety)


def auto_chunk(shape, meta: dict, device: str, n_total: int) -> int:
    """Shots per Deepwave call. GPU: as many as fit in a fraction of the free memory. CPU: small fixed batch."""
    sim = meta.get("sim", {})
    if sim.get("chunk"):
        return max(1, min(int(sim["chunk"]), n_total))
    if is_cuda(device):
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        usable = free * float(sim.get("memory_fraction", 0.7))
        return int(max(1, min(n_total, usable // bytes_per_shot(shape, meta))))
    return min(n_total, 8 if len(shape) == 2 else 2)


def describe(device: str) -> str:
    if not is_cuda(device):
        return f"cpu ({torch.get_num_threads()} threads)"
    d = torch.device(device)
    p = torch.cuda.get_device_properties(d)
    free, total = torch.cuda.mem_get_info(d)
    return (f"{p.name}, compute capability {p.major}.{p.minor}, {total / 2**30:.1f} GB "
            f"({free / 2**30:.1f} GB free), torch {torch.__version__} / CUDA {torch.version.cuda}")


def environment_check(device: str = "auto") -> bool:
    """Print what torch sees and run a tiny Deepwave elastic simulation on the device and on the CPU."""
    import deepwave
    from deepwave.common import vpvsrho_to_lambmubuoyancy

    print(f"torch {torch.__version__}, built with CUDA {torch.version.cuda}, deepwave {deepwave._version.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
    try:
        dev = resolve_device(device)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        return False
    print(f"device: {dev}: {describe(dev)}")

    def run(dv):
        vp = torch.full((80, 60), 5000.0, device=dv)
        lamb, mu, buoy = vpvsrho_to_lambmubuoyancy(vp, vp / 1.75, torch.full_like(vp, 2700.0))
        nt = 400
        amp = deepwave.wavelets.ricker(500.0, nt, 2e-5, 3e-3).to(dv).reshape(1, 1, nt).repeat(2, 1, 1)
        src = torch.tensor([[[40, 30]], [[30, 20]]], dtype=torch.long, device=dv)
        rcv = torch.tensor([[[10, 10], [70, 50]]] * 2, dtype=torch.long, device=dv)
        t0 = time.perf_counter()
        out = deepwave.elastic(lamb, mu, buoy, 0.5, 2e-5, source_amplitudes_y=amp, source_locations_y=src,
                               receiver_locations_y=rcv, pml_width=10, pml_freq=500.0)
        sync(dv)
        return out[-2].cpu().numpy(), time.perf_counter() - t0

    ref, tc = run("cpu")
    if not is_cuda(dev):
        print(f"CPU run OK ({tc:.2f} s). No GPU selected.")
        return True
    try:
        res, tg = run(dev)
    except Exception as e:  # noqa: BLE001 - report anything the GPU path raises
        print(f"FAILED on {dev}: {type(e).__name__}: {e}")
        print("If the message mentions 'no kernel image', Deepwave's CUDA kernels were not built for this GPU "
              "architecture; rebuild Deepwave from source with a CUDA toolkit that supports the card.")
        return False
    err = float(np.abs(res - ref).max() / (np.abs(ref).max() + 1e-30))
    ok = err < 1e-4
    print(f"GPU run {'OK' if ok else 'MISMATCH'}: max relative difference to CPU {err:.1e} "
          f"(GPU {tg:.2f} s incl. start-up, CPU {tc:.2f} s)")
    return ok
