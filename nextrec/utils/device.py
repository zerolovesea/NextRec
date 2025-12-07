"""
Device management utilities for NextRec

Date: create on 03/12/2025
Checkpoint: edit on 06/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch
import platform
import logging


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        mac_ver = platform.mac_ver()[0]
        try:
            major, _ = (int(x) for x in mac_ver.split(".")[:2])
        except Exception:
            major, _ = 0, 0
        if major >= 14:
            return "mps"
    return "cpu"


def get_device_info() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "mps_available": torch.backends.mps.is_available(),
        "current_device": resolve_device(),
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)

    return info


def configure_device(
    distributed: bool, local_rank: int, base_device: torch.device | str = "cpu"
) -> torch.device:
    try:
        device = torch.device(base_device)
    except Exception:
        logging.warning(
            "[configure_device Warning] Invalid base_device, falling back to CPU."
        )
        return torch.device("cpu")

    if distributed:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                logging.warning(
                    "[Distributed Warning] CUDA requested but unavailable. Falling back to CPU."
                )
                return torch.device("cpu")
            if not (0 <= local_rank < torch.cuda.device_count()):
                logging.warning(
                    f"[Distributed Warning] local_rank {local_rank} is invalid for available CUDA devices. Falling back to CPU."
                )
                return torch.device("cpu")
            try:
                torch.cuda.set_device(local_rank)
                return torch.device(f"cuda:{local_rank}")
            except Exception as exc:
                logging.warning(
                    f"[Distributed Warning] Failed to set CUDA device for local_rank {local_rank}: {exc}. Falling back to CPU."
                )
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    return device
