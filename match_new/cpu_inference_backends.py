"""Lazy, FP32-only CPU adapters for an unnormalized HardNet ``q`` graph.

Only ``forward_features(patches).q`` is exported. Native binary models export
that method on the backbone, never the hash head. Normalization, Hadamard,
sign handling and bit packing belong to the caller, not to these backends.

Cache hits need neither optional packages nor an ONNX export. Misses use unique
staging files; a short OS file lock protects the two-file publication/rollback.
A separate process-local lock is necessary because PyTorch's legacy ONNX
exporter has process-global tracing state (it is not thread-safe).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import marshal
import operator
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn


GRAPH_SCHEMA_VERSION = "hardnet_raw_q_v1"
_EXPORT_LOCK = threading.Lock()
_GRAPH_OPTIMIZATION_LEVELS = ("disabled", "basic", "extended", "all")


class _InferenceExportGraph(nn.Module):
    def __init__(self, model: nn.Module, *, native_binary: bool) -> None:
        super().__init__()
        effective_model = getattr(model, "backbone", None) if native_binary else model
        if not isinstance(effective_model, nn.Module) or not callable(
            getattr(effective_model, "forward_features", None)
        ):
            target = "model.backbone" if native_binary else "model"
            raise ValueError(f"{target} must expose forward_features(patches).q.")
        # Register only the actual inference model, not an unused binary head.
        self.features_model = effective_model

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.features_model.forward_features(patches).q


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = int(value.strip(), 10) if isinstance(value, str) else operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}.")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean, got {value!r}.")
    return bool(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _cache_key(metadata: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(metadata)).hexdigest()


def _tensor_signature(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("The FP32 ONNX export cache requires dense, unquantized tensors.")
    # Byte views also support scalars and dtypes that NumPy cannot represent.
    raw = tensor.reshape(-1).view(torch.uint8).numpy()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
    }


def _configuration_value(value: Any) -> Any:
    """Serialize graph-affecting options without unstable object addresses."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, (tuple, list, torch.Size)):
        return [_configuration_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _configuration_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _model_structure(model: nn.Module) -> list[dict[str, Any]]:
    structure = []
    for name, module in model.named_modules():
        options = {
            key: _configuration_value(value)
            for key, value in sorted(vars(module).items())
            if not key.startswith("_") and key != "training" and not callable(value)
        }
        structure.append({
            "name": name,
            "class": f"{type(module).__module__}.{type(module).__qualname__}",
            "options": options,
            "extra_repr": module.extra_repr(),
        })
    return structure


def _inference_model_state_sha256(model: nn.Module, structure: Any) -> str:
    # State_dict alone misses dropout p, BN eps/affine and nonpersistent buffers.
    entries = {name: _configuration_value(value) for name, value in sorted(model.state_dict().items())}
    buffers = {name: _tensor_signature(value) for name, value in model.named_buffers()}
    payload = {"structure": structure, "state_dict": entries, "all_buffers": buffers}
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _callable_signature(function: Any) -> dict[str, Any]:
    function = getattr(function, "__func__", function)
    code = getattr(function, "__code__", None)
    result = {"name": f"{getattr(function, '__module__', '')}.{getattr(function, '__qualname__', type(function).__qualname__)}"}
    if code is not None:
        result["code_sha256"] = hashlib.sha256(marshal.dumps(code)).hexdigest()
        result["defaults"] = _configuration_value(getattr(function, "__defaults__", None))
        result["kwdefaults"] = _configuration_value(getattr(function, "__kwdefaults__", None))
        closure = getattr(function, "__closure__", None)
        if closure:
            result["closure"] = [_configuration_value(cell.cell_contents) for cell in closure]
    return result


def _model_code_sha256(model: nn.Module) -> str:
    files: dict[str, str] = {}
    methods = []
    for name, module in model.named_modules():
        for cls in type(module).__mro__:
            try:
                filename = inspect.getsourcefile(cls)
            except (TypeError, OSError):
                filename = None
            if filename and filename not in files:
                try:
                    files[filename] = _sha256_file(Path(filename))
                except OSError:
                    files[filename] = "source-unavailable"
        for method_name in ("forward", "forward_features"):
            method = getattr(module, method_name, None)
            if callable(method):
                methods.append({"module": name, "method": method_name, **_callable_signature(method)})
    # Whole defining files cover helper functions; bytecode covers runtime overrides.
    payload = {"source_files": files, "methods": methods, "export_forward": _callable_signature(_InferenceExportGraph.forward)}
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _export_snapshot(model: nn.Module, *, native_binary: bool) -> _InferenceExportGraph:
    try:
        snapshot = copy.deepcopy(_InferenceExportGraph(model, native_binary=native_binary))
        snapshot.to(device="cpu", dtype=torch.float32, memory_format=torch.contiguous_format)
        snapshot.eval()
        # .to(memory_format=...) only promises a layout for image tensors. Ensure
        # linear weights and every registered buffer are contiguous as well.
        for parameter in snapshot.parameters():
            parameter.data = parameter.data.contiguous()
        for module in snapshot.modules():
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    module._buffers[name] = buffer.contiguous()
        return snapshot
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Cannot copy the model for CPU FP32 ONNX export: {exc}") from exc


def _cache_matches(onnx_path: Path, manifest_path: Path, metadata: dict[str, Any]) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False
        checksum = manifest.pop("onnx_sha256", None)
        if manifest != {**metadata, "cache_key": _cache_key(metadata)}:
            return False
        return isinstance(checksum, str) and _sha256_file(onnx_path) == checksum
    except (OSError, ValueError, TypeError):
        return False


def _temporary_file(destination: Path, *, suffix: str) -> Path:
    descriptor, filename = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=suffix, dir=str(destination.parent))
    os.close(descriptor)  # Windows exporters/replace need the staging handle closed.
    return Path(filename)


def _remove_temporary(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Do not turn a successful commit into an apparent export failure.
            pass


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _publication_lock(onnx_path: Path) -> Iterator[None]:
    """Lock only cache publication, not hits, model hashing or ONNX export."""
    lock_path = onnx_path.with_suffix(onnx_path.suffix + ".lock")
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _publish_cache(onnx_temp: Path, manifest_temp: Path, onnx_path: Path, manifest_path: Path, metadata: dict[str, Any]) -> None:
    rollback_path: Path | None = None
    with _publication_lock(onnx_path):
        if _cache_matches(onnx_path, manifest_path, metadata):
            return  # Another process completed exactly this graph in the meantime.
        try:
            if manifest_path.exists():
                rollback_path = _temporary_file(manifest_path, suffix=".rollback")
                _write_bytes(rollback_path, manifest_path.read_bytes())
            # Manifest first: if the final ONNX replace fails, the old ONNX was
            # never touched and only this small manifest needs rolling back.
            os.replace(manifest_temp, manifest_path)
            try:
                os.replace(onnx_temp, onnx_path)
            except BaseException:
                if rollback_path is None:
                    manifest_path.unlink(missing_ok=True)
                else:
                    os.replace(rollback_path, manifest_path)
                raise
        finally:
            _remove_temporary(rollback_path)


def _validate_graph_contract(input_shape: list[Any], output_shape: list[Any], *, input_float32: bool, output_float32: bool, source: str) -> tuple[int, int, int]:
    def static_positive(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def dynamic(value: Any) -> bool:
        return value is None or isinstance(value, str)

    if not input_float32 or not output_float32:
        raise ValueError(f"{source} must have float32 input and raw-q output; FP16/BF16 graphs are unsupported.")
    if len(input_shape) != 4 or not dynamic(input_shape[0]) or input_shape[1] != 1:
        raise ValueError(f"{source} input must be NCHW [dynamic_batch,1,H,W], got {input_shape}.")
    if not all(static_positive(value) for value in input_shape[2:]) or input_shape[2] != input_shape[3]:
        raise ValueError(f"{source} input must have fixed, positive square patch dimensions, got {input_shape}.")
    if len(output_shape) != 2 or not dynamic(output_shape[0]) or not static_positive(output_shape[1]):
        raise ValueError(f"{source} raw-q output must be [dynamic_batch,descriptor_dim], got {output_shape}.")
    return input_shape[2], input_shape[3], output_shape[1]


def _onnx_shape(value_info: Any) -> list[Any]:
    return [dimension.dim_value if dimension.HasField("dim_value") else (dimension.dim_param or None) for dimension in value_info.type.tensor_type.shape.dim]


def ensure_onnx_model(
    model: nn.Module,
    *,
    checkpoint_path: str | Path,
    native_binary: bool,
    patch_size: int,
    fused_conv_bn_pairs: int,
    configured_path: str | Path | None = None,
    opset_version: int = 17,
    export_info: dict[str, Any] | None = None,
) -> Path:
    """Return a checksum-verified raw-q graph, or atomically export a new one.

    ``model`` and the checkpoint are read-only. Hashes describe a deep-copied,
    eval-mode CPU FP32 inference model, including effective module options and
    fused weights. A configured filename receives exactly the same validation
    as the default content-addressed cache in the system temporary directory.
    ``export_info`` optionally receives a cache-hit flag and actual tracing/export
    time; cache verification time is not mislabeled as a fresh ONNX export.
    """
    if export_info is not None:
        export_info.update(cache_hit=False, export_ms=0.0)
    patch_size = _integer(patch_size, "patch_size", minimum=1)
    fused_conv_bn_pairs = _integer(fused_conv_bn_pairs, "fused_conv_bn_pairs")
    opset_version = _integer(opset_version, "opset_version", minimum=7)
    native_binary = _boolean(native_binary, "native_binary")
    from torch.onnx._constants import ONNX_MAX_OPSET

    if opset_version > ONNX_MAX_OPSET:
        raise ValueError(f"opset_version={opset_version} exceeds this PyTorch exporter's maximum {ONNX_MAX_OPSET}.")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ONNX export checkpoint file does not exist: {checkpoint}")
    snapshot = _export_snapshot(model, native_binary=native_binary)
    structure = _model_structure(snapshot.features_model)
    metadata = {
        "checkpoint_sha256": _sha256_file(checkpoint),
        "inference_model_state_sha256": _inference_model_state_sha256(snapshot.features_model, structure),
        "model_code_sha256": _model_code_sha256(snapshot.features_model),
        "model_graph_sha256": hashlib.sha256(_json_bytes(structure)).hexdigest(),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_output": "raw_q",
        "graph_kind": f"{type(snapshot.features_model).__module__}.{type(snapshot.features_model).__qualname__}",
        "native_binary": native_binary,
        "patch_size": patch_size,
        "fused_conv_bn_pairs": fused_conv_bn_pairs,
        "opset_version": opset_version,
        "torch_version": str(torch.__version__),
    }
    onnx_path = (
        Path(configured_path).expanduser().resolve()
        if configured_path is not None
        else Path(tempfile.gettempdir()) / "hardnet_onnx_cpu_cache" / f"hardnet_{_cache_key(metadata)}.onnx"
    )
    manifest_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    lock_path = onnx_path.with_suffix(onnx_path.suffix + ".lock")
    if checkpoint in {onnx_path, manifest_path.resolve(), lock_path.resolve()}:
        raise ValueError("The ONNX graph, manifest and lock paths must differ from the read-only checkpoint path.")
    if _cache_matches(onnx_path, manifest_path, metadata):
        if export_info is not None:
            export_info["cache_hit"] = True
        return onnx_path
    try:
        import onnx
    except (ImportError, OSError) as exc:
        raise RuntimeError("ONNX export requires a working 'onnx' package. Install it in the active environment: python -m pip install onnx") from exc

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        onnx_temp = _temporary_file(onnx_path, suffix=".tmp.onnx")
        sample = torch.zeros((1, 1, patch_size, patch_size), dtype=torch.float32)
        with _EXPORT_LOCK:
            if _cache_matches(onnx_path, manifest_path, metadata):
                if export_info is not None:
                    export_info["cache_hit"] = True
                return onnx_path
            export_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
                torch.onnx.export(
                    snapshot,
                    sample,
                    str(onnx_temp),
                    input_names=["patches"],
                    output_names=["features"],
                    dynamic_axes={"patches": {0: "batch"}, "features": {0: "batch"}},
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    external_data=False,
                    dynamo=False,
                )
            if export_info is not None:
                export_info["export_ms"] = (time.perf_counter() - export_started) * 1000.0
        graph = onnx.load(str(onnx_temp), load_external_data=False)
        if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
            raise ValueError("The exported raw-q graph must have exactly one input and one output.")
        graph_input, graph_output = graph.graph.input[0], graph.graph.output[0]
        _validate_graph_contract(
            _onnx_shape(graph_input), _onnx_shape(graph_output),
            input_float32=graph_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT,
            output_float32=graph_output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT,
            source="Exported ONNX graph",
        )
        onnx.helper.set_model_props(graph, {
            "hardnet.graph_schema_version": GRAPH_SCHEMA_VERSION,
            "hardnet.output": "raw_q",
            "hardnet.cache_key": _cache_key(metadata),
        })
        onnx.checker.check_model(graph)
        onnx.save_model(graph, str(onnx_temp), save_as_external_data=False)
        manifest_temp = _temporary_file(manifest_path, suffix=".tmp.json")
        manifest = {**metadata, "cache_key": _cache_key(metadata), "onnx_sha256": _sha256_file(onnx_temp)}
        _write_bytes(manifest_temp, _json_bytes(manifest))
        _publish_cache(onnx_temp, manifest_temp, onnx_path, manifest_path, metadata)
    except Exception as exc:
        raise RuntimeError(f"Failed to export/publish CPU FP32 raw-q ONNX graph at {onnx_path}: {exc}") from exc
    finally:
        _remove_temporary(onnx_temp)
        _remove_temporary(manifest_temp)
    return onnx_path


class _CPUBackend:
    @staticmethod
    def prepare_input(batch: torch.Tensor) -> np.ndarray:
        """Detach/copy to CPU FP32 NCHW; never alter pixel values or normalize."""
        if not isinstance(batch, torch.Tensor):
            raise TypeError("prepare_input expects a torch.Tensor in NCHW layout.")
        if batch.layout != torch.strided or batch.is_complex():
            raise ValueError("prepare_input requires a dense, real-valued tensor.")
        if batch.ndim != 4 or batch.shape[1] != 1 or any(size <= 0 for size in batch.shape):
            raise ValueError(f"prepare_input expects a nonempty NCHW [N,1,H,W] tensor, got {tuple(batch.shape)}.")
        return batch.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()

    def _validate_input(self, batch: np.ndarray) -> None:
        if not isinstance(batch, np.ndarray):
            raise TypeError("CPU backend calls expect a NumPy ndarray; use prepare_input(tensor) first.")
        if batch.dtype != np.float32 or not batch.flags.c_contiguous:
            raise ValueError("CPU backend input must be a contiguous float32 ndarray.")
        if batch.ndim != 4 or batch.shape[0] <= 0 or tuple(batch.shape[1:]) != (1, self.patch_height, self.patch_width):
            raise ValueError(f"CPU backend input must be [N,1,{self.patch_height},{self.patch_width}] with N > 0, got {batch.shape}.")

    def _output_tensor(self, output: Any, batch_size: int) -> torch.Tensor:
        array = np.asarray(output)
        if array.dtype != np.float32 or array.shape != (batch_size, self.descriptor_dim):
            raise RuntimeError(f"CPU backend returned an invalid raw-q tensor: dtype={array.dtype}, shape={array.shape}; expected float32 [{batch_size},{self.descriptor_dim}].")
        # Own the storage for the ONNX Runtime output.
        return torch.from_numpy(array.copy(order="C"))

    def settings(self) -> dict[str, Any]:
        """JSON-serializable initialized settings; requested and actual values differ."""
        return copy.deepcopy(self._settings)

    def metadata(self) -> dict[str, Any]:
        return self.settings()


def _backend_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"CPU backend ONNX graph does not exist: {resolved}")
    return resolved


class OnnxRuntimeCPUBackend(_CPUBackend):
    """ONNX Runtime with CPUExecutionProvider only, without fallback or autocast."""

    def __init__(self, onnx_path: str | Path, *, intra_op_num_threads: int = 0, inter_op_num_threads: int = 1, graph_optimization_level: str = "all") -> None:
        intra_threads = _integer(intra_op_num_threads, "model.backend_num_threads")
        inter_threads = _integer(inter_op_num_threads, "model.backend_inter_op_num_threads")
        level = str(graph_optimization_level).strip().lower()
        if level not in _GRAPH_OPTIMIZATION_LEVELS:
            raise ValueError("model.backend_graph_optimization must be disabled/basic/extended/all.")
        self.onnx_path = _backend_path(onnx_path)
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            raise RuntimeError("ONNX Runtime CPU backend requires a working 'onnxruntime' package. Install it in the active environment: python -m pip install onnxruntime") from exc
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable; no alternative provider will be used.")
        levels = {
            "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_threads
        options.inter_op_num_threads = inter_threads
        options.graph_optimization_level = levels[level]
        try:
            self.session = ort.InferenceSession(str(self.onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
            self.session.disable_fallback()
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize ONNX Runtime CPU FP32 graph {self.onnx_path}: {exc}") from exc
        providers = self.session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError(f"ONNX Runtime initialized unexpected providers: {providers}; only CPUExecutionProvider is permitted.")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ONNX Runtime graph must have exactly one input and one raw-q output.")
        graph_input, graph_output = inputs[0], outputs[0]
        self.patch_height, self.patch_width, self.descriptor_dim = _validate_graph_contract(
            graph_input.shape, graph_output.shape,
            input_float32=graph_input.type == "tensor(float)",
            output_float32=graph_output.type == "tensor(float)",
            source="ONNX Runtime graph",
        )
        self.input_name, self.output_name = graph_input.name, graph_output.name
        actual = self.session.get_session_options()
        props = self.session.get_modelmeta().custom_metadata_map
        self._settings = {
            "backend": "onnxruntime", "device": "CPU", "inference_precision": "fp32",
            "version": str(ort.__version__), "onnx_path": str(self.onnx_path), "providers": providers,
            "backend_num_threads": actual.intra_op_num_threads,
            "backend_inter_op_num_threads": actual.inter_op_num_threads,
            "backend_graph_optimization": level,
            "requested_backend_num_threads": intra_threads,
            "requested_backend_inter_op_num_threads": inter_threads,
            "input_name": self.input_name, "output_name": self.output_name,
            "input_shape": graph_input.shape, "output_shape": graph_output.shape,
            "graph_output": props.get("hardnet.output"),
            "graph_schema_version": props.get("hardnet.graph_schema_version"),
            "cache_key": props.get("hardnet.cache_key"),
        }

    def __call__(self, batch: np.ndarray) -> torch.Tensor:
        self._validate_input(batch)
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        return self._output_tensor(output, batch.shape[0])



def create_cpu_backend(
    name: str,
    path: str | Path,
    config: Mapping[str, Any],
) -> OnnxRuntimeCPUBackend:
    """Create the only supported external backend: ONNX Runtime CPU."""

    backend_name = str(name).strip().lower()
    if backend_name != "onnxruntime":
        raise ValueError(
            "Only ONNX Runtime CPU inference is supported; "
            f"got {backend_name!r}."
        )
    if not isinstance(config, Mapping) or not isinstance(config.get("model", {}), Mapping):
        raise ValueError("CPU backend config and config['model'] must be mappings.")
    model_config = config["model"]
    threads = _integer(model_config.get("backend_num_threads", 0), "model.backend_num_threads")
    inter_threads = _integer(
        model_config.get("backend_inter_op_num_threads", 1),
        "model.backend_inter_op_num_threads",
    )
    graph_level = str(model_config.get("backend_graph_optimization", "all")).strip().lower()
    return OnnxRuntimeCPUBackend(
        path,
        intra_op_num_threads=threads,
        inter_op_num_threads=inter_threads,
        graph_optimization_level=graph_level,
    )
