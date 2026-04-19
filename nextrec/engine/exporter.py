from __future__ import annotations

import inspect
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from nextrec.basic.loggers import colorize
from nextrec.basic.session import get_save_path
from nextrec.utils.onnx_utils import OnnxModelWrapper, create_dummy_inputs
from nextrec.utils.torch_utils import smart_inference_mode

if TYPE_CHECKING:
    from nextrec.engine.model import Model


class Exporter:
    supported_export_formats = {"pt", "onnx", "openvino"}

    def export(
        self: "Model",
        format: str = "pt",
        save_path: str | Path | None = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> Path:
        export_format = str(format).lower()
        if export_format not in self.supported_export_formats:
            raise ValueError(
                f"[NextRec-export Error] Unsupported export format '{format}'. "
                f"Supported formats: {sorted(self.supported_export_formats)}"
            )
        if export_format == "pt":
            return self.save_model(
                save_path=save_path,
                add_timestamp=kwargs.pop("add_timestamp", False),
                verbose=True,
            )
        if export_format == "onnx":
            return self.export_onnx(save_path=save_path, batch_size=batch_size)
        raise NotImplementedError("[NextRec-export Error] openvino export is not implemented yet.")

    @smart_inference_mode()
    def export_onnx(
        self: "Model",
        save_path: str | Path | None = None,
        batch_size: int = 1,
    ) -> Path:
        model_to_export = self
        model_to_export = model_to_export.to(self.device)
        model_to_export.eval()

        input_names = [feat.name for feat in self.all_features]
        dummy_inputs = create_dummy_inputs(
            self.all_features,
            batch_size=batch_size,
            device=self.device,
        )
        wrapper = OnnxModelWrapper(model_to_export, input_names)
        with torch.no_grad():
            output_sample = wrapper(*dummy_inputs)
        if isinstance(output_sample, (tuple, list)):
            output_names = [f"output_{idx}" for idx in range(len(output_sample))]
        else:
            output_names = ["output"]
        target_path = get_save_path(
            path=save_path,
            default_dir=self.session.root,
            default_name=f"{self.model_name}_onnx",
            suffix="onnx",
        )
        export_kwargs: dict[str, Any] = {}
        export_sig = inspect.signature(torch.onnx.export)
        if "dynamo" in export_sig.parameters:
            export_kwargs["dynamo"] = True

        with redirect_stdout(io.StringIO()):
            torch.onnx.export(
                wrapper,
                tuple(dummy_inputs),
                target_path,
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=18,
                do_constant_folding=True,
                **export_kwargs,
            )

        logging.info(colorize(f"ONNX model exported to: {target_path}", color="green"))
        return Path(target_path)
