import torch
import platform

def resolve_device() -> str:
    """Select a usable device with graceful fallback."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        mac_ver = platform.mac_ver()[0]
        try:
            major, minor = (int(x) for x in mac_ver.split(".")[:2])
        except Exception:
            major, minor = 0, 0
        if major >= 14:
            return "mps"
    return "cpu"