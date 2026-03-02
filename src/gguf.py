"""GGUF conversion and quantization tools wrapping llama.cpp utilities."""

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Generator
from urllib import request
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).parent.parent / "tools" / "llama_cpp"
MODELS_DIR = Path(__file__).parent.parent / "models"

GITHUB_API_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
CONVERT_SCRIPT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"

# Output types supported by convert_hf_to_gguf.py
CONVERT_OUT_TYPES = ["auto", "f32", "f16", "bf16", "q8_0"]

# Recommended quant types (shown by default)
RECOMMENDED_QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16"]

# All quant types supported by llama-quantize
ALL_QUANT_TYPES = [
    "Q2_K", "Q2_K_S",
    "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "F16", "BF16", "F32",
    "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_NL", "IQ4_XS",
    "COPY",
]


# ---------------------------------------------------------------------------
# Tool status
# ---------------------------------------------------------------------------

def get_tool_status() -> dict:
    """Check whether llama.cpp tools are installed and return status info."""
    quantize_exe = TOOLS_DIR / "llama-quantize.exe"
    convert_script = TOOLS_DIR / "convert_hf_to_gguf.py"

    quantize_available = quantize_exe.exists()
    convert_available = convert_script.exists()

    version = None
    if quantize_available or convert_available:
        # Try to detect version from directory name or release info
        release_info = TOOLS_DIR / "_release_info.json"
        if release_info.exists():
            try:
                with open(release_info, "r") as f:
                    info = json.load(f)
                version = info.get("tag_name", "unknown")
            except (json.JSONDecodeError, OSError):
                version = "installed"
        else:
            version = "installed"

    return {
        "quantize_available": quantize_available,
        "convert_available": convert_available,
        "version": version,
    }


# ---------------------------------------------------------------------------
# Download tools
# ---------------------------------------------------------------------------

def download_llama_cpp_tools() -> Generator[str, None, None]:
    """Download llama.cpp tools from GitHub. Yields progress strings."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Find latest release ---
    yield "Fetching latest llama.cpp release info..."
    try:
        req = request.Request(GITHUB_API_LATEST, headers={"User-Agent": "HereticConverter"})
        with request.urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError, OSError) as e:
        yield f"ERROR: Failed to fetch release info: {e}"
        return

    tag = release.get("tag_name", "unknown")
    yield f"Latest release: {tag}"

    # --- Step 2: Find the Windows CPU x64 zip asset ---
    assets = release.get("assets", [])
    zip_asset = None
    for asset in assets:
        name = asset.get("name", "")
        if "win" in name and "cpu" in name and "x64" in name and name.endswith(".zip"):
            zip_asset = asset
            break

    if not zip_asset:
        # Fallback: try any Windows zip
        for asset in assets:
            name = asset.get("name", "")
            if "win" in name and name.endswith(".zip") and "cuda" not in name.lower():
                zip_asset = asset
                break

    if not zip_asset:
        yield "ERROR: Could not find Windows x64 binary in the latest release."
        yield "Available assets: " + ", ".join(a["name"] for a in assets[:10])
        return

    zip_url = zip_asset["browser_download_url"]
    zip_size = zip_asset.get("size", 0)
    zip_name = zip_asset["name"]
    size_mb = zip_size / (1024 * 1024) if zip_size else 0

    yield f"Downloading {zip_name} ({size_mb:.1f} MB)..."

    # --- Step 3: Download the zip ---
    zip_path = TOOLS_DIR / zip_name
    try:
        _download_with_progress(zip_url, zip_path, zip_size, progress_callback=None)
    except Exception as e:
        yield f"ERROR: Download failed: {e}"
        return

    yield f"Downloaded {zip_name}"

    # --- Step 4: Extract the zip ---
    yield "Extracting..."
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract all files, flattening the top-level directory
            top_dirs = set()
            for info in zf.infolist():
                parts = info.filename.split("/")
                if len(parts) > 1:
                    top_dirs.add(parts[0])

            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = info.filename.split("/")
                # Strip top-level directory if all files share one
                if len(top_dirs) == 1 and len(parts) > 1:
                    out_name = "/".join(parts[1:])
                else:
                    out_name = info.filename

                if not out_name:
                    continue

                out_path = TOOLS_DIR / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

        zip_path.unlink(missing_ok=True)
    except Exception as e:
        yield f"ERROR: Extraction failed: {e}"
        return

    yield "Extracted llama.cpp binaries"

    # --- Step 5: Download convert_hf_to_gguf.py ---
    yield "Downloading convert_hf_to_gguf.py..."
    convert_path = TOOLS_DIR / "convert_hf_to_gguf.py"
    try:
        req = request.Request(CONVERT_SCRIPT_URL, headers={"User-Agent": "HereticConverter"})
        with request.urlopen(req, timeout=30) as resp:
            convert_path.write_bytes(resp.read())
    except (URLError, OSError) as e:
        yield f"ERROR: Failed to download convert script: {e}"
        return

    yield "Downloaded convert_hf_to_gguf.py"

    # --- Step 6: Save release info ---
    try:
        with open(TOOLS_DIR / "_release_info.json", "w") as f:
            json.dump({"tag_name": tag, "asset": zip_name}, f, indent=2)
    except OSError:
        pass

    # Verify
    status = get_tool_status()
    if status["quantize_available"] and status["convert_available"]:
        yield f"All tools installed successfully ({tag})"
    else:
        missing = []
        if not status["quantize_available"]:
            missing.append("llama-quantize.exe")
        if not status["convert_available"]:
            missing.append("convert_hf_to_gguf.py")
        yield f"WARNING: Some tools not found after install: {', '.join(missing)}"


def _download_with_progress(url: str, dest: Path, total_size: int, progress_callback=None):
    """Download a file from a URL to disk."""
    req = request.Request(url, headers={"User-Agent": "HereticConverter"})
    with request.urlopen(req, timeout=120) as resp:
        with open(dest, "wb") as f:
            downloaded = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size:
                    progress_callback(downloaded, total_size)


# ---------------------------------------------------------------------------
# Convert HF to GGUF
# ---------------------------------------------------------------------------

def convert_hf_to_gguf(
    model_dir: str,
    output_path: str,
    out_type: str = "auto",
) -> Generator[tuple[str, str], None, None]:
    """Convert a HuggingFace model directory to GGUF format.

    Yields (dashboard_html, status_text) tuples for real-time progress.
    """
    convert_script = TOOLS_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        yield ("", "ERROR: convert_hf_to_gguf.py not found. Please download tools first.")
        return

    model_dir = str(model_dir).strip()
    if not model_dir or not Path(model_dir).is_dir():
        yield ("", f"ERROR: Invalid model directory: {model_dir}")
        return

    if not (Path(model_dir) / "config.json").exists():
        yield ("", f"ERROR: No config.json found in {model_dir}. Not a valid HuggingFace model.")
        return

    output_path = str(output_path).strip()
    if not output_path:
        yield ("", "ERROR: Output path is empty.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(convert_script),
        model_dir,
        "--outfile", output_path,
        "--outtype", out_type,
    ]

    yield (
        _make_gguf_dashboard_html(phase="Converting", detail=f"Type: {out_type}"),
        f"Starting conversion...\n  Source: {model_dir}\n  Output: {output_path}\n  Type: {out_type}",
    )

    env = os.environ.copy()
    env["NO_LOCAL_GGUF"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(TOOLS_DIR),
        )

        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                # Keep last 30 lines for display
                display_lines = lines[-30:]
                yield (
                    _make_gguf_dashboard_html(phase="Converting", detail=f"Type: {out_type}"),
                    "\n".join(display_lines),
                )

        proc.wait()

        if proc.returncode == 0:
            out = Path(output_path)
            size_str = _format_file_size(out.stat().st_size) if out.exists() else "unknown"
            yield (
                _make_gguf_dashboard_html(phase="Complete", detail=f"{out.name} ({size_str})", success=True),
                f"Conversion complete!\n  Output: {output_path}\n  Size: {size_str}\n\n" + "\n".join(lines[-10:]),
            )
        else:
            yield (
                _make_gguf_dashboard_html(phase="Failed", detail="See log below", success=False),
                f"Conversion failed (exit code {proc.returncode}):\n\n" + "\n".join(lines[-20:]),
            )

    except FileNotFoundError:
        yield ("", f"ERROR: Python interpreter not found: {sys.executable}")
    except Exception as e:
        yield ("", f"ERROR: Conversion failed: {e}")


# ---------------------------------------------------------------------------
# Quantize GGUF
# ---------------------------------------------------------------------------

def quantize_gguf(
    input_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    n_threads: int = 0,
    allow_requantize: bool = False,
    leave_output_tensor: bool = False,
    pure: bool = False,
) -> Generator[tuple[str, str], None, None]:
    """Quantize a GGUF file to a different quantization level.

    Yields (dashboard_html, status_text) tuples for real-time progress.
    """
    quantize_exe = TOOLS_DIR / "llama-quantize.exe"
    if not quantize_exe.exists():
        yield ("", "ERROR: llama-quantize.exe not found. Please download tools first.")
        return

    input_path = str(input_path).strip()
    if not input_path or not Path(input_path).is_file():
        yield ("", f"ERROR: Input file not found: {input_path}")
        return

    output_path = str(output_path).strip()
    if not output_path:
        yield ("", "ERROR: Output path is empty.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    input_size = Path(input_path).stat().st_size
    input_size_str = _format_file_size(input_size)

    cmd = [str(quantize_exe)]

    if allow_requantize:
        cmd.append("--allow-requantize")
    if leave_output_tensor:
        cmd.append("--leave-output-tensor")
    if pure:
        cmd.append("--pure")
    if n_threads > 0:
        cmd.extend(["--nthread", str(n_threads)])

    cmd.extend([input_path, output_path, quant_type])

    yield (
        _make_gguf_dashboard_html(
            phase="Quantizing",
            detail=f"{quant_type} | Input: {input_size_str}",
        ),
        f"Starting quantization...\n  Input: {input_path} ({input_size_str})\n  Output: {output_path}\n  Type: {quant_type}",
    )

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(TOOLS_DIR),
        )

        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                display_lines = lines[-30:]
                yield (
                    _make_gguf_dashboard_html(
                        phase="Quantizing",
                        detail=f"{quant_type} | Input: {input_size_str}",
                    ),
                    "\n".join(display_lines),
                )

        proc.wait()

        if proc.returncode == 0:
            out = Path(output_path)
            if out.exists():
                out_size = out.stat().st_size
                out_size_str = _format_file_size(out_size)
                ratio = out_size / input_size if input_size > 0 else 0
                yield (
                    _make_gguf_dashboard_html(
                        phase="Complete",
                        detail=f"{out.name} ({out_size_str}, {ratio:.1%} of original)",
                        success=True,
                    ),
                    f"Quantization complete!\n  Output: {output_path}\n  Size: {out_size_str} ({ratio:.1%} of original)\n\n"
                    + "\n".join(lines[-10:]),
                )
            else:
                yield (
                    _make_gguf_dashboard_html(phase="Complete", detail=quant_type, success=True),
                    f"Quantization complete!\n  Output: {output_path}\n\n" + "\n".join(lines[-10:]),
                )
        else:
            yield (
                _make_gguf_dashboard_html(phase="Failed", detail="See log below", success=False),
                f"Quantization failed (exit code {proc.returncode}):\n\n" + "\n".join(lines[-20:]),
            )

    except FileNotFoundError:
        yield ("", f"ERROR: llama-quantize.exe not found at {quantize_exe}")
    except Exception as e:
        yield ("", f"ERROR: Quantization failed: {e}")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_gguf_files() -> list[str]:
    """Scan ./models/ for *.gguf files."""
    found = []
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.rglob("*.gguf")):
            found.append(str(p.resolve()))
    return found


def discover_hf_model_dirs() -> list[str]:
    """Scan ./models/ and HF cache for directories with config.json + safetensors."""
    found = []

    # App models directory
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.iterdir()):
            if p.is_dir() and (p / "config.json").exists():
                has_safetensors = any(p.glob("*.safetensors"))
                if has_safetensors:
                    found.append(str(p.resolve()))

    # HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        for model_dir in sorted(hf_cache.glob("models--*")):
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                snaps = sorted(
                    snapshots.iterdir(),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for snap in snaps:
                    if (snap / "config.json").exists() and any(snap.glob("*.safetensors")):
                        found.append(str(snap.resolve()))
                        break

    return found


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

def _make_gguf_dashboard_html(
    phase: str = "",
    detail: str = "",
    success: bool | None = None,
) -> str:
    """Build a status dashboard HTML matching the dark zinc/emerald style."""
    if not phase:
        return ""

    if success is True:
        icon = '<span style="color:#10b981;font-size:1.3em;">&#10003;</span>'
        phase_color = "#10b981"
    elif success is False:
        icon = '<span style="color:#ef4444;font-size:1.3em;">&#10007;</span>'
        phase_color = "#ef4444"
    else:
        icon = '<span style="color:#eab308;font-size:1.3em;">&#9679;</span>'
        phase_color = "#eab308"

    return (
        '<div style="display:flex;gap:1rem;padding:0.75rem;background:#18181b;'
        'border:1px solid #27272a;border-radius:8px;font-family:\'Geist Mono\',monospace;'
        'align-items:center;">'
        f'<div style="flex:0 0 auto;">{icon}</div>'
        '<div style="flex:1;">'
        f'<div style="color:{phase_color};font-size:1.1em;font-weight:bold;">{phase}</div>'
        f'<div style="color:#a1a1aa;font-size:0.85em;margin-top:0.15rem;">{detail}</div>'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def make_lm_studio_path(
    gguf_filename: str,
    source_path: str = "",
) -> str:
    """Build an LM-Studio-compatible output path under MODELS_DIR.

    LM Studio expects:  <models_dir>/<publisher>/<model-name>/<file>.gguf
    This function infers publisher and model-name from the source, falling
    back to "local" as the publisher when it can't be determined.

    Returns the full absolute path string to the .gguf output file.
    """
    publisher, model_name = _parse_publisher_model(source_path)

    # The model folder gets a -GGUF suffix if it doesn't already have one
    if not model_name.upper().endswith("-GGUF"):
        model_folder = f"{model_name}-GGUF"
    else:
        model_folder = model_name

    out_dir = MODELS_DIR / publisher / model_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / gguf_filename)


def _parse_publisher_model(source_path: str) -> tuple[str, str]:
    """Try to extract (publisher, model_name) from a source path.

    Handles:
    - HF cache paths: .../models--Qwen--Qwen3-4B/snapshots/abc/
    - HF identifiers: Qwen/Qwen3-4B-Instruct
    - Local model dirs: ./models/Qwen3-4B-decensored
    - GGUF file paths: ./models/local/SomeModel-GGUF/SomeModel-F16.gguf
    """
    source_path = str(source_path).strip()

    # 1) HF cache pattern: models--<publisher>--<model>
    match = re.search(r"models--([^/\\]+)--([^/\\]+)", source_path)
    if match:
        return match.group(1), match.group(2)

    # 2) Already in our LM Studio layout: .../models/<publisher>/<model-name>/...
    #    Check if the path is under MODELS_DIR with two levels
    try:
        rel = Path(source_path).resolve().relative_to(MODELS_DIR.resolve())
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0], parts[1]
    except (ValueError, OSError):
        pass

    # 3) HF-style "org/model" string (e.g. typed into the HF model path field)
    if "/" in source_path and not source_path.startswith((".", "/", "\\")) and ":" not in source_path:
        segments = source_path.strip("/").split("/")
        if len(segments) == 2 and segments[0] and segments[1]:
            return segments[0], segments[1]

    # 4) Fallback: publisher = "local", model = directory or file basename
    p = Path(source_path)
    name = p.stem if p.suffix.lower() == ".gguf" else p.name
    if not name:
        name = "model"
    return "local", strip_quant_suffix(name)


def strip_quant_suffix(name: str) -> str:
    """Strip existing quantization suffix from a model/file name.

    E.g., 'model-Q4_K_M' -> 'model', 'model-F16.gguf' -> 'model'
    """
    # Remove .gguf extension first
    if name.lower().endswith(".gguf"):
        name = name[:-5]

    # Strip known quant type suffixes (case-insensitive)
    all_types_lower = [t.lower() for t in ALL_QUANT_TYPES]
    # Try removing a trailing -TYPE or _TYPE or .TYPE
    for sep in ["-", "_", "."]:
        parts = name.rsplit(sep, 1)
        if len(parts) == 2 and parts[1].lower().replace("-", "_") in all_types_lower:
            name = parts[0]
            break

    return name
