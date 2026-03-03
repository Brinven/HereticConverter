"""Helper utilities for the Heretic Converter."""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Union


def get_model_size(path: Union[str, Path]) -> int:
    """Calculate total size of a model directory in bytes."""
    total = 0
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m:02d}m {s:02d}s"


def validate_model_dir(path: Union[str, Path]) -> bool:
    """Check if a directory contains model files (config.json or .safetensors)."""
    path = Path(path)
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = any(path.glob("*.safetensors"))
    has_gguf = any(path.glob("*.gguf"))
    return has_config or has_weights or has_gguf


def list_saved_models(output_dir: str = "./models") -> list:
    """List all saved models in the output directory."""
    models = []
    output_path = Path(output_dir)

    if not output_path.exists():
        return models

    for dirpath, dirnames, filenames in os.walk(output_path):
        item = Path(dirpath)

        # Skip the root models/ dir itself — we only want its children
        if item == output_path:
            # But pick up any loose GGUF files sitting directly in models/
            for f in filenames:
                if f.endswith(".gguf"):
                    fpath = item / f
                    size = fpath.stat().st_size
                    models.append({
                        "name": f,
                        "path": str(fpath),
                        "size": format_size(size),
                        "size_bytes": size,
                    })
            continue

        has_config = "config.json" in filenames
        has_weights = any(
            f.endswith(".safetensors") or f.endswith(".npz") or f.endswith(".gguf")
            for f in filenames
        )

        if has_config or has_weights:
            size = get_model_size(item)
            models.append({
                "name": item.name,
                "path": str(item),
                "size": format_size(size),
                "size_bytes": size,
            })
            dirnames.clear()

    return sorted(models, key=lambda m: m["name"])


def zip_model(model_path: str) -> dict:
    """Zip a model directory for download/transfer."""
    model_dir = Path(model_path)
    if not model_dir.exists() or not model_dir.is_dir():
        return {
            "success": False,
            "message": f"Model directory not found: {model_path}",
            "zip_path": "",
        }

    files = list(model_dir.iterdir())
    has_model_files = any(
        f.name == "config.json" or f.suffix in (".safetensors", ".npz", ".gguf")
        for f in files
        if f.is_file()
    )
    if not has_model_files:
        return {
            "success": False,
            "message": "Directory doesn't contain model files (config.json or weights).",
            "zip_path": "",
        }

    try:
        tmp_dir = tempfile.mkdtemp()
        zip_base = os.path.join(tmp_dir, model_dir.name)
        zip_path = shutil.make_archive(
            zip_base, "zip",
            root_dir=str(model_dir.parent),
            base_dir=model_dir.name,
        )
        size_str = format_size(os.path.getsize(zip_path))
        return {
            "success": True,
            "message": f"Zipped {model_dir.name} ({size_str})",
            "zip_path": zip_path,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to create zip: {e}",
            "zip_path": "",
        }


def import_model_zip(zip_path: str, output_dir: str = "./models") -> dict:
    """Import a model from an uploaded zip file."""
    if not zip_path or not os.path.isfile(zip_path):
        return {"success": False, "message": "No file provided.", "model_path": ""}

    if not zipfile.is_zipfile(zip_path):
        return {
            "success": False,
            "message": "File is not a valid zip archive.",
            "model_path": "",
        }

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            has_model_files = any(
                n.endswith("config.json")
                or n.endswith(".safetensors")
                or n.endswith(".npz")
                or n.endswith(".gguf")
                for n in names
            )
            if not has_model_files:
                return {
                    "success": False,
                    "message": "Invalid model zip. Archive must contain config.json or .safetensors files.",
                    "model_path": "",
                }

            top_dirs = {n.split("/")[0] for n in names if "/" in n}

            os.makedirs(output_dir, exist_ok=True)

            if len(top_dirs) == 1:
                model_name = top_dirs.pop()
                dest = os.path.join(output_dir, model_name)
                if os.path.exists(dest):
                    return {
                        "success": False,
                        "message": f"Model '{model_name}' already exists. Delete it first or rename.",
                        "model_path": "",
                    }
                zf.extractall(output_dir)
            else:
                model_name = Path(zip_path).stem
                dest = os.path.join(output_dir, model_name)
                if os.path.exists(dest):
                    return {
                        "success": False,
                        "message": f"Model '{model_name}' already exists. Delete it first or rename.",
                        "model_path": "",
                    }
                os.makedirs(dest)
                zf.extractall(dest)

            size = get_model_size(dest)
            return {
                "success": True,
                "message": f"Imported '{model_name}' ({format_size(size)})",
                "model_path": dest,
            }

    except zipfile.BadZipFile:
        return {
            "success": False,
            "message": "Corrupted zip file.",
            "model_path": "",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Import failed: {e}",
            "model_path": "",
        }


def check_cuda_available() -> tuple[bool, str]:
    """Check if CUDA is available and return device info."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "No CUDA GPU detected"

        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory
        vram_str = format_size(vram)
        return True, f"{device_name} ({vram_str} VRAM)"
    except ImportError:
        return False, "PyTorch is not installed"
    except Exception as e:
        return False, f"CUDA check failed: {e}"
