import logging
import numpy as np
import torch
import torch.distributed as dist


def configure_device(distributed: bool, local_rank: int, base_device: torch.device | str = "cpu") -> torch.device:
    device = torch.device(base_device)
    if distributed and device.type == "cuda":
        if not torch.cuda.is_available():
            logging.warning("[Distributed Warning] CUDA device requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        try:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        except Exception as exc:
            logging.warning(f"[Distributed Warning] Failed to set CUDA device for local_rank {local_rank}: {exc}. Using CPU instead.")
            device = torch.device("cpu")
    return device

def init_process_group(distributed: bool, rank: int, world_size: int, device_id: int | None = None) -> None:
    """
    initialize distributed process group for multi-GPU training.
    
    Args:
        distributed: whether to enable distributed training
        rank: global rank of the current process
        world_size: total number of processes
        device_id: CUDA device id for nccl backend (to avoid barrier warnings)
    """
    if not distributed or not dist.is_available() or dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # for nccl need specify device_id to avoid barrier warnings
    if backend == "nccl" and device_id is not None:
        dist.init_process_group(backend=backend,  init_method='env://',  rank=rank,  world_size=world_size, device_id=torch.device(f'cuda:{device_id}'))
    else:
        dist.init_process_group(backend=backend,  init_method='env://',  rank=rank,  world_size=world_size)

def gather_numpy(self, array: np.ndarray | None) -> np.ndarray | None:
    if array is None or not (self.distributed and dist.is_available() and dist.is_initialized()):
        return array
    world_size = dist.get_world_size()
    # tensor = torch.tensor(array, device=self.device)
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).to(self.device)
    else:
        tensor = torch.tensor(array, device=self.device)
    local_len = torch.tensor([tensor.shape[0]], device=self.device, dtype=torch.long)
    lengths = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    max_len = int(max(l.item() for l in lengths))
    if tensor.numel() == 0:
        pad_shape = (max_len,) + tuple(tensor.shape[1:])
        padded = torch.zeros(pad_shape, device=self.device, dtype=tensor.dtype)
    else:
        pad_len = max_len - tensor.shape[0]
        if pad_len > 0:
            pad_shape = (pad_len,) + tuple(tensor.shape[1:])
            pad_tensor = torch.zeros(pad_shape, device=self.device, dtype=tensor.dtype)
            padded = torch.cat([tensor, pad_tensor], dim=0)
        else:
            padded = tensor
    gather_list = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gather_list, padded)
    pieces = []
    for idx, gathered in enumerate(gather_list):
        length = lengths[idx].item()
        if length > 0:
            pieces.append(gathered[:length])
    if not pieces:
        return None
    return torch.cat(pieces, dim=0).cpu().numpy()
