"""Public exports for multitask models."""

from nextrec.models.multitask.aitm import AITM
from nextrec.models.multitask.apg import APG
from nextrec.models.multitask.base import BaseMultitaskModel
from nextrec.models.multitask.cross_stitch import CrossStitch
from nextrec.models.multitask.esmm import ESMM
from nextrec.models.multitask.hmoe import HMOE
from nextrec.models.multitask.mmoe import MMOE
from nextrec.models.multitask.pepnet import PEPNet
from nextrec.models.multitask.ple import PLE
from nextrec.models.multitask.poso import POSO
from nextrec.models.multitask.share_bottom import ShareBottom
from nextrec.models.multitask.snr_trans import SNRTrans

__all__ = [
    "AITM",
    "APG",
    "BaseMultitaskModel",
    "CrossStitch",
    "ESMM",
    "HMOE",
    "MMOE",
    "PEPNet",
    "PLE",
    "POSO",
    "SNRTrans",
    "ShareBottom",
]
