"""Офлайн-конвертация локального HF snapshot GigaChat из block FP8 в BF16.

Поддерживается только публичный формат источника: model_type=deepseek_v3,
quant_method=fp8, activation_scheme=dynamic, weight_block_size=[128, 128],
веса float8_e4m3fn и парные *.weight_scale_inv float32.

Конвертация идёт по одному выходному шарду на CPU. Сеть, CUDA и remote code не
используются. Выходная папка публикуется одним переименованием только после
полного успеха; при исключении временная папка-сосед удаляется.

Запуск:
    py -3 convert_gigachat_fp8.py --input <snapshot> --output <bf16_snapshot>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
BLOCK_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn
FP8_DTYPE_NAME = "F8_E4M3"
SCALE_DTYPE = torch.float32
SCALE_DTYPE_NAME = "F32"
SCALE_SUFFIX = "_scale_inv"
WEIGHT_SUFFIX = ".weight"

# Только известные tokenizer/generation-файлы. Код модели и remote code не копируются.
METADATA_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
)


class ConvertError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]


def _read_json(path: Path, code: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise ConvertError(code, f"файл не найден: {path}") from exc
    except (OSError, ValueError) as exc:
        raise ConvertError(code, f"не удалось прочитать {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConvertError(code, f"{path}: ожидался JSON-объект")
    return data


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _validate_quant_config(config: dict) -> None:
    if config.get("model_type") != "deepseek_v3":
        raise ConvertError(
            "unsupported_model",
            f"ожидается model_type=deepseek_v3, получено {config.get('model_type')!r}",
        )
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        raise ConvertError("unsupported_config", "в config.json нет объекта quantization_config")
    if qc.get("quant_method") != "fp8":
        raise ConvertError(
            "unsupported_config",
            f"ожидается quantization_config.quant_method=fp8, получено {qc.get('quant_method')!r}",
        )
    if qc.get("activation_scheme") != "dynamic":
        raise ConvertError(
            "unsupported_config",
            "ожидается quantization_config.activation_scheme=dynamic, "
            f"получено {qc.get('activation_scheme')!r}",
        )
    block = qc.get("weight_block_size")
    if not isinstance(block, (list, tuple)) or list(block) != [BLOCK_SIZE, BLOCK_SIZE]:
        raise ConvertError(
            "unsupported_config",
            f"ожидается weight_block_size=[{BLOCK_SIZE}, {BLOCK_SIZE}], получено {block!r}",
        )


def _validate_shard_name(name: object) -> str:
    """Имя шарда из индекса — только простой файл .safetensors внутри snapshot.

    Проверка намеренно лексическая: обычный HF snapshot может держать файл
    символической ссылкой в blob cache за пределами папки snapshot.
    """
    if not isinstance(name, str) or not name:
        raise ConvertError("invalid_shard_path", f"некорректное имя шарда в индексе: {name!r}")
    if name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
        raise ConvertError("invalid_shard_path", f"имя шарда выходит за пределы snapshot: {name!r}")
    path = Path(name)
    if path.is_absolute() or path.name != name:
        raise ConvertError("invalid_shard_path", f"имя шарда выходит за пределы snapshot: {name!r}")
    if not name.endswith(".safetensors"):
        raise ConvertError("invalid_shard_path", f"ожидается файл .safetensors: {name!r}")
    return name


def _expected_scale_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) < 2:
        raise ConvertError("unsupported_weight_shape", f"FP8-вес должен быть >= 2D, получено {shape}")
    rows = (shape[-2] + BLOCK_SIZE - 1) // BLOCK_SIZE
    cols = (shape[-1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    return tuple(shape[:-2]) + (rows, cols)


def _scan_snapshot(input_dir: Path, weight_map: dict) -> tuple[dict[str, TensorInfo], dict[str, list[str]]]:
    shard_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name:
            raise ConvertError("invalid_index", f"некорректное имя тензора в индексе: {name!r}")
        shard = _validate_shard_name(shard)
        shard_names.setdefault(shard, []).append(name)

    infos: dict[str, TensorInfo] = {}
    for shard, names in shard_names.items():
        path = input_dir / shard
        if not path.is_file():
            raise ConvertError("shard_missing", f"шард из индекса не найден: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            expected = set(names)
            if keys != expected:
                missing = sorted(expected - keys)
                extra = sorted(keys - expected)
                raise ConvertError(
                    "index_shard_mismatch",
                    f"{shard}: индекс и файл не совпадают; нет в файле={missing[:5]}, "
                    f"лишние в файле={extra[:5]}",
                )
            for name in names:
                sl = handle.get_slice(name)
                infos[name] = TensorInfo(
                    name=name,
                    shard=shard,
                    dtype=str(sl.get_dtype()),
                    shape=tuple(int(dim) for dim in sl.get_shape()),
                )
    return infos, shard_names


def _build_plan(infos: dict[str, TensorInfo]) -> tuple[dict[str, str], set[str]]:
    fp8_weights: list[str] = []
    scale_names: set[str] = set()
    for info in infos.values():
        if info.dtype.startswith("F8_"):
            if info.dtype != FP8_DTYPE_NAME:
                raise ConvertError(
                    "unsupported_fp8_dtype",
                    f"{info.name}: поддерживается только {FP8_DTYPE_NAME}, получено {info.dtype}",
                )
            if not info.name.endswith(WEIGHT_SUFFIX):
                raise ConvertError(
                    "fp8_outside_weight",
                    f"FP8-тензор не является ожидаемым *.weight: {info.name}",
                )
            fp8_weights.append(info.name)
        if info.name.endswith(SCALE_SUFFIX):
            scale_names.add(info.name)

    weight_to_scale: dict[str, str] = {}
    for name in fp8_weights:
        scale_name = name + SCALE_SUFFIX
        if scale_name not in infos:
            raise ConvertError("missing_scale", f"для FP8-веса нет парной scale: {name}")
        weight_to_scale[name] = scale_name

    expected_scales = set(weight_to_scale.values())
    orphan = sorted(scale_names - expected_scales)
    if orphan:
        raise ConvertError("orphan_scale", f"scale без парного FP8-веса: {orphan[:5]}")
    if not fp8_weights:
        raise ConvertError("no_fp8_weights", "во входном snapshot нет FP8-весов для конвертации")

    for name, scale_name in weight_to_scale.items():
        weight = infos[name]
        scale = infos[scale_name]
        if scale.dtype != SCALE_DTYPE_NAME:
            raise ConvertError(
                "scale_dtype_mismatch",
                f"{scale_name}: ожидается {SCALE_DTYPE_NAME}, получено {scale.dtype}",
            )
        expected = _expected_scale_shape(weight.shape)
        if scale.shape != expected:
            raise ConvertError(
                "scale_shape_mismatch",
                f"{scale_name}: ожидается shape {expected}, получено {scale.shape}",
            )
    return weight_to_scale, expected_scales


def dequantize_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Block FP8 E4M3FN → BF16: умножение блоков в FP32, затем один каст в BF16."""
    if weight.dtype != FP8_DTYPE:
        raise ConvertError("weight_dtype_mismatch", f"ожидается {FP8_DTYPE}, получено {weight.dtype}")
    if scale.dtype != SCALE_DTYPE:
        raise ConvertError("scale_dtype_mismatch", f"ожидается {SCALE_DTYPE}, получено {scale.dtype}")
    shape = tuple(int(dim) for dim in weight.shape)
    expected = _expected_scale_shape(shape)
    if tuple(int(dim) for dim in scale.shape) != expected:
        raise ConvertError(
            "scale_shape_mismatch",
            f"ожидается scale shape {expected}, получено {tuple(scale.shape)}",
        )

    rows, cols = shape[-2], shape[-1]
    row_blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty(shape, dtype=torch.bfloat16)
    scale_f32 = scale.to(torch.float32)
    for block_row in range(row_blocks):
        r0 = block_row * BLOCK_SIZE
        r1 = min(r0 + BLOCK_SIZE, rows)
        # Один ряд блоков: (..., r1-r0, cols) в FP32 и сетка scale по столбцам.
        weight_strip = weight[..., r0:r1, :].to(torch.float32)
        strip_scale = scale_f32[..., block_row, :].repeat_interleave(BLOCK_SIZE, dim=-1)[..., :cols]
        out[..., r0:r1, :] = (weight_strip * strip_scale.unsqueeze(-2)).to(torch.bfloat16)
    return out


def save_shard(tensors: dict[str, torch.Tensor], path: Path) -> None:
    save_file(tensors, str(path), metadata={"format": "pt"})


def _convert_shard(
    input_dir: Path,
    output_path: Path,
    shard: str,
    names: list[str],
    infos: dict[str, TensorInfo],
    weight_to_scale: dict[str, str],
    scale_shards: dict[str, str],
    save_fn: Callable[[dict[str, torch.Tensor], Path], None],
) -> int:
    """Сконвертировать один выходной шард. Все handles закрываются до возврата."""
    tensors: dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        source = stack.enter_context(safe_open(str(input_dir / shard), framework="pt", device="cpu"))
        scale_handles: dict[str, object] = {}
        for name in names:
            info = infos[name]
            tensor = source.get_tensor(name)
            if name in weight_to_scale:
                scale_name = weight_to_scale[name]
                scale_shard = scale_shards[scale_name]
                if scale_shard == shard:
                    scale_handle = source
                else:
                    scale_handle = scale_handles.get(scale_shard)
                    if scale_handle is None:
                        scale_handle = stack.enter_context(
                            safe_open(str(input_dir / scale_shard), framework="pt", device="cpu")
                        )
                        scale_handles[scale_shard] = scale_handle
                scale = scale_handle.get_tensor(scale_name)
                if tensor.dtype != FP8_DTYPE:
                    raise ConvertError(
                        "weight_dtype_mismatch",
                        f"{name}: ожидается {FP8_DTYPE}, получено {tensor.dtype}",
                    )
                if tuple(tensor.shape) != info.shape:
                    raise ConvertError(
                        "weight_shape_mismatch",
                        f"{name}: ожидается shape {info.shape}, получено {tuple(tensor.shape)}",
                    )
                limited = infos[scale_name]
                if tuple(scale.shape) != limited.shape:
                    raise ConvertError(
                        "scale_shape_mismatch",
                        f"{scale_name}: ожидается shape {limited.shape}, получено {tuple(scale.shape)}",
                    )
                tensors[name] = dequantize_blockwise(tensor, scale)
                del tensor, scale
            else:
                if tensor.dtype == FP8_DTYPE:
                    raise ConvertError(
                        "fp8_outside_weight",
                        f"FP8-тензор не является ожидаемым *.weight: {name}",
                    )
                tensors[name] = tensor

        save_fn(tensors, output_path)
        total_size = sum(int(t.numel()) * int(t.element_size()) for t in tensors.values())
    return total_size


def _safe_rmtree(path: Path) -> None:
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            gc.collect()
            if attempt < 4:
                time.sleep(0.2)
            else:
                shutil.rmtree(path, ignore_errors=True)


def _prepare_paths(input_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    source = Path(input_path).expanduser().resolve()
    if not source.is_dir():
        raise ConvertError("input_missing", f"входная папка не найдена: {source}")
    raw_output = Path(output_path).expanduser()
    if raw_output.exists() or raw_output.is_symlink():
        raise ConvertError("output_exists", f"выходная папка уже существует: {raw_output}")
    output = raw_output.resolve()
    if output.exists() or output.is_symlink():
        raise ConvertError("output_exists", f"выходная папка уже существует: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ConvertError("output_conflicts_input", "выходная папка не должна пересекаться с источником")
    return source, output


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    save_fn: Callable[[dict[str, torch.Tensor], Path], None] = save_shard,
) -> dict:
    source, output = _prepare_paths(input_path, output_path)

    config = _read_json(source / CONFIG_NAME, "config_missing")
    _validate_quant_config(config)
    index = _read_json(source / INDEX_NAME, "index_missing")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConvertError("invalid_index", "в model.safetensors.index.json нет непустого weight_map")

    infos, shard_names = _scan_snapshot(source, weight_map)
    weight_to_scale, consumed_scales = _build_plan(infos)

    scale_shards = {name: infos[name].shard for name in infos if name.endswith(SCALE_SUFFIX)}
    output_names_by_shard: dict[str, list[str]] = {shard: [] for shard in shard_names}
    for name, info in infos.items():
        if name in consumed_scales:
            continue
        output_names_by_shard[info.shard].append(name)

    output_config = dict(config)
    output_config.pop("quantization_config", None)
    for key in ("torch_dtype", "dtype"):
        if isinstance(output_config.get(key), str):
            output_config[key] = "bfloat16"

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    published = False
    staging_created = False
    total_size = 0
    converted = 0
    preserved = 0
    copied: list[str] = []
    try:
        staging.mkdir()
        staging_created = True
        for shard in shard_names:
            names = output_names_by_shard[shard]
            total_size += _convert_shard(
                source,
                staging / shard,
                shard,
                names,
                infos,
                weight_to_scale,
                scale_shards,
                save_fn,
            )
            converted += sum(1 for name in names if name in weight_to_scale)
            preserved += sum(1 for name in names if name not in weight_to_scale)

        output_weight_map = {
            name: info.shard
            for name, info in infos.items()
            if name not in consumed_scales
        }
        metadata = dict(index.get("metadata") or {})
        metadata["total_size"] = total_size
        output_index = {"metadata": metadata, "weight_map": output_weight_map}

        _write_json(staging / CONFIG_NAME, output_config)
        _write_json(staging / INDEX_NAME, output_index)
        for name in METADATA_FILES:
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, staging / name)
                copied.append(name)

        if output.exists() or output.is_symlink():
            raise ConvertError("output_exists", f"выходная папка создана во время конвертации: {output}")
        os.replace(staging, output)
        published = True
    finally:
        if not published and staging_created:
            _safe_rmtree(staging)

    return {
        "input": str(source),
        "output": str(output),
        "shards": len(shard_names),
        "converted_weights": converted,
        "removed_scales": len(consumed_scales),
        "preserved_tensors": preserved,
        "total_size": total_size,
        "copied_files": copied,
    }


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="Офлайн CPU-конвертер GigaChat block FP8 E4M3FN → BF16"
    )
    parser.add_argument("--input", required=True, help="локальная папка HF snapshot (config + index + shards)")
    parser.add_argument("--output", required=True, help="новая папка для BF16 snapshot")
    args = parser.parse_args(argv)
    try:
        summary = convert(args.input, args.output)
    except ConvertError as exc:
        print(f"convert: {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"convert: unexpected_error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "convert: ok "
        f"shards={summary['shards']} converted={summary['converted_weights']} "
        f"removed_scales={summary['removed_scales']} preserved={summary['preserved_tensors']} "
        f"total_size={summary['total_size']} out={summary['output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
