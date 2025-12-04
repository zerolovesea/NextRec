"""
Device management utilities for NextRec

Date: create on 03/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""
import os
import torch
import platform
import multiprocessing


def resolve_device() -> str:
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

def get_device_info() -> dict:
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': torch.backends.mps.is_available(),
        'current_device': resolve_device(),
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_capability'] = torch.cuda.get_device_capability(0)
    
    return info
