"""Heretic Converter — A Gradio app for decensoring and de-slopping language models."""

from pathlib import Path

import gradio as gr
from gradio.themes import Base
from gradio.themes.utils import colors, sizes, fonts

from src.config import PRESETS, DEFAULTS, load_config, save_config
from src.decensor import run_decensor
from src.evaluator import evaluate_model, chat_with_model
from src.gguf import (
    get_tool_status, download_llama_cpp_tools,
    convert_hf_to_gguf, quantize_gguf,
    discover_gguf_files, discover_hf_model_dirs,
    strip_quant_suffix, make_lm_studio_path,
    CONVERT_OUT_TYPES, RECOMMENDED_QUANT_TYPES, ALL_QUANT_TYPES,
    MODELS_DIR,
)
from src.utils import list_saved_models, zip_model, import_model_zip

DEFAULT_OUTPUT_DIR = "./models"

# --- Dark zinc/emerald theme ---

_zinc = colors.Color(
    c50="#fafafa", c100="#f4f4f5", c200="#e4e4e7", c300="#d4d4d8",
    c400="#a1a1aa", c500="#71717a", c600="#52525b", c700="#3f3f46",
    c800="#27272a", c900="#18181b", c950="#09090b",
    name="zinc",
)

_emerald = colors.Color(
    c50="#ecfdf5", c100="#d1fae5", c200="#a7f3d0", c300="#6ee7b7",
    c400="#34d399", c500="#10b981", c600="#059669", c700="#047857",
    c800="#065f46", c900="#064e3b", c950="#022c22",
    name="emerald",
)

dark_theme = Base(
    primary_hue=_emerald,
    secondary_hue=_zinc,
    neutral_hue=_zinc,
    font=fonts.GoogleFont("Geist"),
    font_mono=fonts.GoogleFont("Geist Mono"),
).set(
    body_background_fill="#09090b",
    body_background_fill_dark="#09090b",
    body_text_color="#f4f4f5",
    body_text_color_dark="#f4f4f5",
    body_text_color_subdued="#a1a1aa",
    body_text_color_subdued_dark="#a1a1aa",
    background_fill_primary="#18181b",
    background_fill_primary_dark="#18181b",
    background_fill_secondary="#27272a",
    background_fill_secondary_dark="#27272a",
    block_background_fill="#18181b",
    block_background_fill_dark="#18181b",
    block_border_color="#1e1e22",
    block_border_color_dark="#1e1e22",
    block_label_background_fill="#27272a",
    block_label_background_fill_dark="#27272a",
    block_label_text_color="#a1a1aa",
    block_label_text_color_dark="#a1a1aa",
    block_title_text_color="#f4f4f5",
    block_title_text_color_dark="#f4f4f5",
    input_background_fill="#27272a",
    input_background_fill_dark="#27272a",
    input_border_color="#27272a",
    input_border_color_dark="#27272a",
    input_placeholder_color="#71717a",
    input_placeholder_color_dark="#71717a",
    border_color_accent="#10b981",
    border_color_accent_dark="#10b981",
    border_color_primary="#1e1e22",
    border_color_primary_dark="#1e1e22",
    link_text_color="#34d399",
    link_text_color_dark="#34d399",
    link_text_color_hover="#6ee7b7",
    link_text_color_hover_dark="#6ee7b7",
    button_primary_background_fill="#10b981",
    button_primary_background_fill_dark="#10b981",
    button_primary_background_fill_hover="#059669",
    button_primary_background_fill_hover_dark="#059669",
    button_primary_text_color="#022c22",
    button_primary_text_color_dark="#022c22",
    button_secondary_background_fill="#27272a",
    button_secondary_background_fill_dark="#27272a",
    button_secondary_background_fill_hover="#3f3f46",
    button_secondary_background_fill_hover_dark="#3f3f46",
    button_secondary_text_color="#f4f4f5",
    button_secondary_text_color_dark="#f4f4f5",
    button_secondary_border_color="#27272a",
    button_secondary_border_color_dark="#27272a",
    shadow_drop="none",
    shadow_drop_lg="none",
    checkbox_background_color="#27272a",
    checkbox_background_color_dark="#27272a",
    checkbox_border_color="#27272a",
    checkbox_border_color_dark="#27272a",
    checkbox_label_background_fill="#18181b",
    checkbox_label_background_fill_dark="#18181b",
    slider_color="#10b981",
    slider_color_dark="#10b981",
    table_even_background_fill="#18181b",
    table_even_background_fill_dark="#18181b",
    table_odd_background_fill="#27272a",
    table_odd_background_fill_dark="#27272a",
    panel_background_fill="#09090b",
    panel_background_fill_dark="#09090b",
    panel_border_color="#1e1e22",
    panel_border_color_dark="#1e1e22",
)

# --- Custom CSS ---

css = """
.main-header { text-align: center; margin-bottom: 0.5em; position: relative; }
.main-header h1 { margin-bottom: 0.1em; color: #f4f4f5; font-family: 'Geist', system-ui, sans-serif; }
.main-header p { color: #a1a1aa; font-size: 0.95em; font-family: 'Geist', system-ui, sans-serif; }
.hub-back {
    position: absolute; top: 0; left: 0;
    color: #a1a1aa; font-size: 0.85em; text-decoration: none;
    font-family: 'Geist', system-ui, sans-serif;
    transition: color 0.15s;
}
.hub-back:hover { color: #34d399; }

.tab-nav button {
    color: #a1a1aa !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Geist', system-ui, sans-serif !important;
    transition: color 0.15s, border-color 0.15s;
}
.tab-nav button.selected {
    color: #f4f4f5 !important;
    border-bottom-color: #10b981 !important;
}
.tab-nav button:hover {
    color: #f4f4f5 !important;
}

button.primary:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
    outline: 2px solid #10b981 !important;
    outline-offset: 2px;
}

textarea[data-testid], .output-textbox textarea {
    font-family: 'Geist Mono', ui-monospace, monospace !important;
}

footer { display: none !important; }

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    padding: 0 2rem !important;
}

.gradio-container .block {
    border-radius: 12px !important;
}

.gradio-container .tabs > .tab-nav {
    margin-bottom: 1rem;
}

.gradio-container .tabitem {
    padding-top: 0.5rem;
}

.gradio-container .form > .block {
    margin-bottom: 0.25rem;
}

input[data-testid="textbox"], .gradio-container textarea:not([rows]) {
    overflow: hidden !important;
}

.accordion { border-color: #1e1e22 !important; }
"""


def _login_hf(token: str) -> None:
    """Log in to HuggingFace Hub if a token is provided."""
    if token and token.strip():
        from huggingface_hub import login
        login(token=token.strip(), add_to_git_credential=False)


# --- Decensor handler (generator) ---

def handle_decensor(
    model_path, output_name, mode, n_trials, quantization, batch_size, max_response_length,
    kl_ceiling, refusal_target, target_output_layers_only, layer_range_min, layer_range_max,
    preset, hf_token,
):
    """Handle the decensor button click."""
    # Coerce gr.Number fields that can return None when cleared
    n_trials = int(n_trials) if n_trials else 450
    batch_size = int(batch_size) if batch_size is not None else 0
    max_response_length = int(max_response_length) if max_response_length else 100
    kl_ceiling = float(kl_ceiling) if kl_ceiling else 1.0
    refusal_target = int(refusal_target) if refusal_target is not None else 5
    layer_range_min = int(layer_range_min or 0)
    layer_range_max = int(layer_range_max or 0)
    model_path = str(model_path).strip() if model_path else ""
    output_name = str(output_name).strip() if output_name else ""
    quantization = quantization or "None"
    mode = mode or "Decensor"

    # Persist settings for next session
    save_config({
        "preset": preset or "Custom",
        "kl_ceiling": kl_ceiling,
        "n_trials": n_trials,
        "refusal_target": refusal_target,
        "target_output_layers_only": bool(target_output_layers_only),
        "layer_range_min": layer_range_min,
        "layer_range_max": layer_range_max,
        "quantization": quantization,
        "batch_size": batch_size,
        "max_response_length": max_response_length,
        "mode": mode,
    })

    try:
        _login_hf(hf_token)
    except Exception as e:
        yield ("", f"HuggingFace login failed: {e}")
        return
    yield from run_decensor(
        model_path=model_path,
        output_name=output_name,
        mode=mode,
        n_trials=n_trials,
        quantization=quantization,
        batch_size=batch_size,
        max_response_length=max_response_length,
        output_dir=DEFAULT_OUTPUT_DIR,
        kl_ceiling=kl_ceiling,
        refusal_target=refusal_target,
        target_output_layers_only=bool(target_output_layers_only),
        layer_range_min=layer_range_min,
        layer_range_max=layer_range_max,
    )


# --- Evaluate handler (generator) ---

def handle_evaluate(model_path, quantization, hf_token):
    """Handle the evaluate button click."""
    try:
        _login_hf(hf_token)
    except Exception as e:
        yield f"HuggingFace login failed: {e}"
        return
    yield from evaluate_model(
        model_path=model_path,
        quantization=quantization,
    )


# --- Chat handler (blocking) ---

def handle_chat(model_path, prompt, max_tokens, quantization, hf_token):
    """Handle the chat button click."""
    try:
        _login_hf(hf_token)
    except Exception as e:
        return f"HuggingFace login failed: {e}"
    return chat_with_model(
        model_path=model_path,
        prompt=prompt,
        max_tokens=int(max_tokens),
        quantization=quantization,
    )


# --- Model management helpers ---

def get_model_choices():
    """Get list of saved model paths for the dropdown."""
    models = list_saved_models(DEFAULT_OUTPUT_DIR)
    return [m["path"] for m in models]


def get_model_names():
    """Get list of saved model names for the selector dropdown."""
    models = list_saved_models(DEFAULT_OUTPUT_DIR)
    return [m["name"] for m in models]


def refresh_model_list():
    """Refresh the list of saved models."""
    models = list_saved_models(DEFAULT_OUTPUT_DIR)
    if not models:
        return "No saved models found in ./models/"
    lines = []
    for m in models:
        lines.append(f"**{m['name']}** — {m['size']}\n  `{m['path']}`")
    return "\n\n".join(lines)


# --- Download/Upload handlers ---

def handle_download(model_name):
    """Handle the download button click."""
    if not model_name:
        return gr.update(), "Please select a model to download."

    models = list_saved_models(DEFAULT_OUTPUT_DIR)
    match = next((m for m in models if m["name"] == model_name), None)
    if not match:
        return gr.update(), f"Model '{model_name}' not found."

    result = zip_model(match["path"])
    if result["success"]:
        return result["zip_path"], result["message"]
    else:
        return gr.update(), result["message"]


def discover_local_models():
    """Find model directories in ./models/ and HuggingFace cache."""
    found = []

    # App output directory
    models_dir = Path(DEFAULT_OUTPUT_DIR)
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and (p / "config.json").exists():
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
                    if (snap / "config.json").exists():
                        found.append(str(snap.resolve()))
                        break  # only latest snapshot per model

    return found


def handle_upload(file):
    """Handle the upload/import of a model zip."""
    if file is None:
        return "No file uploaded."

    file_path = file.name if hasattr(file, "name") else str(file)
    result = import_model_zip(file_path, DEFAULT_OUTPUT_DIR)
    if result["success"]:
        return result["message"]
    else:
        return result["message"]


# --- GGUF tab helpers ---

def _format_tool_status_html(status: dict) -> str:
    """Build a colored banner showing llama.cpp tool install state."""
    q = status["quantize_available"]
    c = status["convert_available"]
    ver = status.get("version")

    if q and c:
        bg, border, text_color = "#022c22", "#065f46", "#6ee7b7"
        icon = "&#10003;"
        label = f"llama.cpp tools installed ({ver})" if ver else "llama.cpp tools installed"
    elif q or c:
        bg, border, text_color = "#422006", "#92400e", "#fbbf24"
        icon = "&#9888;"
        parts = []
        if not q:
            parts.append("llama-quantize.exe missing")
        if not c:
            parts.append("convert_hf_to_gguf.py missing")
        label = "Partial install: " + ", ".join(parts)
    else:
        bg, border, text_color = "#18181b", "#27272a", "#a1a1aa"
        icon = "&#9679;"
        label = "llama.cpp tools not installed — click Download to install"

    return (
        f'<div style="padding:0.6rem 0.9rem;background:{bg};border:1px solid {border};'
        f'border-radius:8px;color:{text_color};font-size:0.9em;font-family:\'Geist\',system-ui,sans-serif;'
        f'display:flex;align-items:center;gap:0.5rem;">'
        f'<span style="font-size:1.1em;">{icon}</span> {label}'
        f'</div>'
    )


def handle_download_tools():
    """Download llama.cpp tools, yielding (status_html, log_text)."""
    log_lines = []
    for msg in download_llama_cpp_tools():
        log_lines.append(msg)
        status = get_tool_status()
        yield (
            _format_tool_status_html(status),
            "\n".join(log_lines),
        )
    # Final status
    status = get_tool_status()
    yield (_format_tool_status_html(status), "\n".join(log_lines))


def handle_convert_to_gguf(model_dir, out_type, output_name):
    """Handle HF-to-GGUF conversion. Saves in LM Studio folder layout."""
    model_dir = str(model_dir).strip() if model_dir else ""
    out_type = out_type or "auto"
    output_name = str(output_name).strip() if output_name else ""

    if not model_dir:
        yield ("", "Please select a model directory.")
        return

    # Build the GGUF filename
    if not output_name:
        dir_name = Path(model_dir).name
        type_label = out_type.upper() if out_type != "auto" else "F16"
        output_name = f"{dir_name}-{type_label}.gguf"

    if not output_name.lower().endswith(".gguf"):
        output_name += ".gguf"

    # Place in LM Studio layout: models/<publisher>/<model-GGUF>/<file>.gguf
    output_path = make_lm_studio_path(output_name, source_path=model_dir)

    yield from convert_hf_to_gguf(model_dir, output_path, out_type)


def handle_quantize_gguf(input_file, quant_type, output_name, n_threads, allow_requantize, leave_output_tensor, pure):
    """Handle GGUF quantization. Saves in LM Studio folder layout."""
    input_file = str(input_file).strip() if input_file else ""
    quant_type = quant_type or "Q4_K_M"
    output_name = str(output_name).strip() if output_name else ""
    n_threads = int(n_threads) if n_threads else 0
    allow_requantize = bool(allow_requantize)
    leave_output_tensor = bool(leave_output_tensor)
    pure = bool(pure)

    if not input_file:
        yield ("", "Please select an input GGUF file.")
        return

    # Build the GGUF filename
    if not output_name:
        base = strip_quant_suffix(Path(input_file).name)
        output_name = f"{base}-{quant_type}.gguf"

    if not output_name.lower().endswith(".gguf"):
        output_name += ".gguf"

    # Place in LM Studio layout: models/<publisher>/<model-GGUF>/<file>.gguf
    output_path = make_lm_studio_path(output_name, source_path=input_file)

    yield from quantize_gguf(input_file, output_path, quant_type, n_threads, allow_requantize, leave_output_tensor, pure)


# --- Preset and reactive UI helpers ---

def apply_preset(preset_name):
    """Update controls from a preset profile selection."""
    preset = PRESETS.get(preset_name, {})
    warning = ""
    if preset_name == "Aggressive":
        warning = (
            '<div style="padding:0.5rem 0.75rem;background:#450a0a;border:1px solid #991b1b;'
            'border-radius:6px;color:#fca5a5;font-size:0.85em;">'
            'Aggressive settings increase risk of model damage. Monitor KL divergence closely.</div>'
        )
    if preset_name == "Custom":
        return [gr.update(), gr.update(), gr.update(), warning]
    return [
        gr.update(value=preset.get("kl_ceiling", DEFAULTS["kl_ceiling"])),
        gr.update(value=preset.get("refusal_target", DEFAULTS["refusal_target"])),
        gr.update(value=preset.get("target_output_layers_only", DEFAULTS["target_output_layers_only"])),
        warning,
    ]


def update_refusal_warning(target):
    """Show warning when refusal target is dangerously low."""
    if target is not None and target < 3:
        return (
            '<div style="padding:0.5rem 0.75rem;background:#422006;border:1px solid #92400e;'
            'border-radius:6px;color:#fbbf24;font-size:0.85em;margin-top:0.25rem;">'
            'Near-zero targets increase model damage risk</div>'
        )
    return ""


def update_quant_reminder(model_path):
    """Show quantization warning or GGUF error if model path suggests issues."""
    path_lower = str(model_path).lower() if model_path else ""
    if "gguf" in path_lower or path_lower.endswith(".gguf"):
        return (
            '<div style="padding:0.75rem 1rem;background:#450a0a;border:1px solid #991b1b;'
            'border-radius:6px;color:#fca5a5;font-size:0.9em;margin-top:0.5rem;">'
            '<strong>GGUF models are not supported.</strong><br>'
            'Abliteration requires full-precision safetensors weights with individual layer access. '
            'GGUF is a single-file quantized format that cannot be modified layer-by-layer.<br><br>'
            'Use the <strong>base HuggingFace repo</strong> instead '
            '(e.g. <code>org/ModelName</code> not <code>org/ModelName-GGUF</code>).</div>'
        )
    if path_lower and any(q in path_lower for q in ["-q4", "-q3", "_q4", "_q3", ".q4", ".q3", "/q4", "/q3"]):
        return (
            '<div style="padding:0.5rem 0.75rem;background:#1e1b4b;border:1px solid #4338ca;'
            'border-radius:6px;color:#a5b4fc;font-size:0.85em;margin-top:0.5rem;">'
            'Lower quantization reduces optimization precision. Q8 or F16 recommended.</div>'
        )
    return ""


def toggle_layer_range(checked):
    """Hide/show custom layer range inputs based on output-layers-only checkbox."""
    return [gr.update(visible=not checked), gr.update(visible=not checked)]


# --- Build Gradio UI ---

def create_app():
    """Create and return the Gradio app."""
    cfg = load_config()

    with gr.Blocks(title="Heretic Converter", theme=dark_theme, css=css) as app:
        gr.HTML(
            """
            <div class="main-header">
                <a class="hub-back"
                   onclick="window.location.href='http://'+window.location.hostname+':9000';return false;"
                   href="#"
                >&larr; Home Hub</a>
                <h1>Heretic Converter</h1>
                <p>Decensor and de-slop language models, then transfer to your Mac for MLX conversion</p>
            </div>
            """
        )

        with gr.Accordion("HuggingFace Token (required for gated models)", open=False):
            gr.Markdown(
                "Some models require access approval. "
                "Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)"
            )
            hf_token = gr.Textbox(
                label="HuggingFace Token",
                placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                type="password",
                info="Saved in this session only — not stored to disk",
            )

        with gr.Tabs():
            # === Decensor Tab ===
            with gr.Tab("Decensor", id="decensor"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=0):
                        model_source = gr.Radio(
                            choices=["HuggingFace", "Local Path"],
                            value="HuggingFace",
                            label="Model Source",
                        )
                        decensor_model_path = gr.Dropdown(
                            label="HuggingFace Model Path",
                            allow_custom_value=True,
                            info="Model identifier from huggingface.co",
                        )
                        gr.HTML(
                            '<div style="padding:0.4rem 0.6rem;background:#18181b;border:1px solid #27272a;'
                            'border-radius:6px;color:#71717a;font-size:0.8em;margin-top:-0.25rem;margin-bottom:0.25rem;">'
                            'Requires safetensors models — GGUF and quantized formats are not supported</div>'
                        )
                        decensor_output_name = gr.Textbox(
                            label="Output Name (optional)",
                            placeholder="Leave blank for auto-generated name",
                            info="Name for the output model directory",
                        )

                    with gr.Column(scale=2, min_width=0):
                        preset_profile = gr.Dropdown(
                            choices=list(PRESETS.keys()),
                            value=cfg.get("preset", "Balanced"),
                            label="Preset Profile",
                            info="Quick-select common configurations",
                        )
                        decensor_n_trials = gr.Number(
                            value=cfg.get("n_trials", 450),
                            minimum=100,
                            maximum=2000,
                            step=10,
                            label="Trial Count",
                            info="More trials = better results but slower (100-2000)",
                        )
                        decensor_quantization = gr.Dropdown(
                            choices=["None", "4-bit (bitsandbytes)"],
                            value=cfg.get("quantization", "None"),
                            label="Quantization",
                            info="4-bit reduces VRAM usage but may affect quality",
                        )

                    with gr.Column(scale=2, min_width=0):
                        decensor_mode = gr.Radio(
                            choices=["Decensor", "De-slop"],
                            value="Decensor",
                            label="Mode",
                        )
                        decensor_batch_size = gr.Slider(
                            minimum=0, maximum=128, value=0, step=1,
                            label="Batch Size",
                            info="0 = auto-detect optimal batch size",
                        )
                        decensor_max_response = gr.Slider(
                            minimum=50, maximum=500, value=100, step=10,
                            label="Max Response Length",
                            info="Tokens generated per evaluation response",
                        )

                # --- Advanced Settings ---
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row(equal_height=True):
                        with gr.Column(min_width=0):
                            kl_ceiling = gr.Number(
                                value=cfg.get("kl_ceiling", 1.0),
                                minimum=0.1,
                                maximum=5.0,
                                step=0.1,
                                label="KL Divergence Ceiling",
                                info="Hard cap — trials exceeding this value are discarded",
                            )
                            refusal_target = gr.Slider(
                                minimum=0, maximum=20,
                                value=cfg.get("refusal_target", 5),
                                step=1,
                                label="Acceptable Refusals per 100",
                                info="Target refusals per 100 prompts",
                            )
                            refusal_warning = gr.HTML("")

                        with gr.Column(min_width=0):
                            target_output_layers = gr.Checkbox(
                                value=cfg.get("target_output_layers_only", True),
                                label="Target output layers only (recommended)",
                                info="Restrict modifications to later transformer layers",
                            )
                            layer_range_min = gr.Number(
                                value=cfg.get("layer_range_min", 0),
                                minimum=0,
                                maximum=200,
                                step=1,
                                label="Layer Range Min",
                                info="Custom start layer (0 = auto)",
                                visible=not cfg.get("target_output_layers_only", True),
                            )
                            layer_range_max = gr.Number(
                                value=cfg.get("layer_range_max", 0),
                                minimum=0,
                                maximum=200,
                                step=1,
                                label="Layer Range Max",
                                info="Custom end layer (0 = auto)",
                                visible=not cfg.get("target_output_layers_only", True),
                            )

                # --- Preset warning banner ---
                preset_warning = gr.HTML("")

                # --- Quantization reminder banner ---
                quant_reminder = gr.HTML("")

                model_source.change(
                    fn=lambda src: gr.update(
                        label="HuggingFace Model Path" if src == "HuggingFace" else "Local Model",
                        info="Model identifier from huggingface.co" if src == "HuggingFace" else "Select a model or paste a path",
                        choices=[] if src == "HuggingFace" else discover_local_models(),
                        value="",
                    ),
                    inputs=model_source,
                    outputs=decensor_model_path,
                )

                decensor_btn = gr.Button(
                    "Start Decensoring", variant="primary", size="lg"
                )

                # --- Results Dashboard ---
                kl_dashboard = gr.HTML("")

                decensor_status = gr.Textbox(
                    label="Status",
                    lines=18,
                    interactive=False,
                )

                decensor_btn.click(
                    fn=handle_decensor,
                    inputs=[
                        decensor_model_path,
                        decensor_output_name,
                        decensor_mode,
                        decensor_n_trials,
                        decensor_quantization,
                        decensor_batch_size,
                        decensor_max_response,
                        kl_ceiling,
                        refusal_target,
                        target_output_layers,
                        layer_range_min,
                        layer_range_max,
                        preset_profile,
                        hf_token,
                    ],
                    outputs=[kl_dashboard, decensor_status],
                )

                # --- Reactive event wiring ---
                preset_profile.change(
                    fn=apply_preset,
                    inputs=preset_profile,
                    outputs=[kl_ceiling, refusal_target, target_output_layers, preset_warning],
                )
                refusal_target.change(
                    fn=update_refusal_warning,
                    inputs=refusal_target,
                    outputs=refusal_warning,
                )
                decensor_model_path.change(
                    fn=update_quant_reminder,
                    inputs=decensor_model_path,
                    outputs=quant_reminder,
                )
                target_output_layers.change(
                    fn=toggle_layer_range,
                    inputs=target_output_layers,
                    outputs=[layer_range_min, layer_range_max],
                )

            # === Evaluate Tab ===
            with gr.Tab("Evaluate", id="evaluate"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=0):
                        with gr.Row():
                            eval_model_path = gr.Dropdown(
                                choices=get_model_choices(),
                                label="Model",
                                allow_custom_value=True,
                                info="Select a decensored model or enter a HuggingFace path",
                                scale=4,
                            )
                            eval_refresh_btn = gr.Button(
                                "Refresh", size="sm", scale=0, min_width=50
                            )

                    with gr.Column(scale=2, min_width=0):
                        eval_prompt = gr.Textbox(
                            label="Test Prompt (optional)",
                            placeholder="Enter a prompt to test, or leave blank for automated evaluation",
                            lines=3,
                        )

                    with gr.Column(scale=2, min_width=0):
                        eval_max_tokens = gr.Slider(
                            minimum=50, maximum=500, value=100, step=10,
                            label="Max Tokens",
                        )
                        eval_quantization = gr.Dropdown(
                            choices=["None", "4-bit (bitsandbytes)"],
                            value="None",
                            label="Quantization",
                        )

                with gr.Row():
                    eval_btn = gr.Button(
                        "Run Evaluation", variant="primary", size="lg"
                    )
                    chat_btn = gr.Button("Test Chat", size="lg")

                eval_output = gr.Textbox(
                    label="Results",
                    lines=10,
                    interactive=False,
                )

                eval_refresh_btn.click(
                    fn=lambda: gr.update(choices=get_model_choices()),
                    outputs=eval_model_path,
                )
                eval_btn.click(
                    fn=handle_evaluate,
                    inputs=[eval_model_path, eval_quantization, hf_token],
                    outputs=eval_output,
                )
                chat_btn.click(
                    fn=handle_chat,
                    inputs=[
                        eval_model_path,
                        eval_prompt,
                        eval_max_tokens,
                        eval_quantization,
                        hf_token,
                    ],
                    outputs=eval_output,
                )

            # === Models Tab ===
            with gr.Tab("Models", id="models"):
                models_display = gr.Markdown(
                    "Click Refresh to see saved models."
                )
                with gr.Row():
                    models_refresh_btn = gr.Button(
                        "Refresh", size="sm", scale=0, min_width=50
                    )
                    model_selector = gr.Dropdown(
                        choices=get_model_names(),
                        label="Select Model",
                        scale=4,
                    )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=0):
                        download_btn = gr.Button(
                            "Download as Zip", variant="primary"
                        )
                        download_status = gr.Textbox(
                            label="Download Status", lines=2, interactive=False
                        )
                        download_file = gr.File(
                            label="Download", interactive=False
                        )

                    with gr.Column(scale=1, min_width=0):
                        upload_file = gr.File(
                            label="Upload Model Zip", file_types=[".zip"]
                        )
                        import_status = gr.Textbox(
                            label="Import Status", lines=2, interactive=False
                        )

                models_refresh_btn.click(
                    fn=lambda: (
                        refresh_model_list(),
                        gr.update(choices=get_model_names()),
                    ),
                    outputs=[models_display, model_selector],
                )
                download_btn.click(
                    fn=handle_download,
                    inputs=model_selector,
                    outputs=[download_file, download_status],
                )
                upload_file.change(
                    fn=handle_upload,
                    inputs=upload_file,
                    outputs=import_status,
                )

            # === GGUF Tab ===
            with gr.Tab("GGUF", id="gguf"):
                # --- Tool status banner ---
                tool_status_html = gr.HTML(
                    _format_tool_status_html(get_tool_status())
                )

                with gr.Row():
                    download_tools_btn = gr.Button(
                        "Download llama.cpp Tools", variant="secondary", size="sm",
                    )
                download_tools_log = gr.Textbox(
                    label="Download Log", lines=6, interactive=False, visible=False,
                )

                download_tools_btn.click(
                    fn=handle_download_tools,
                    outputs=[tool_status_html, download_tools_log],
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=download_tools_log,
                )

                with gr.Tabs():
                    # --- Convert HF to GGUF sub-tab ---
                    with gr.Tab("Convert HF to GGUF"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=3, min_width=0):
                                convert_model_dir = gr.Dropdown(
                                    choices=discover_hf_model_dirs(),
                                    label="Model Directory",
                                    allow_custom_value=True,
                                    info="Select a local safetensors model or paste a path",
                                )
                                with gr.Row():
                                    convert_refresh_btn = gr.Button(
                                        "Refresh", size="sm", scale=0, min_width=50,
                                    )

                            with gr.Column(scale=2, min_width=0):
                                convert_out_type = gr.Dropdown(
                                    choices=CONVERT_OUT_TYPES,
                                    value="auto",
                                    label="Output Type",
                                    info="Precision for the GGUF output (auto detects from model)",
                                )
                                convert_output_name = gr.Textbox(
                                    label="Output Filename (optional)",
                                    placeholder="Auto-generated from model name",
                                    info="Saved to ./models/",
                                )

                        convert_btn = gr.Button(
                            "Convert to GGUF", variant="primary", size="lg",
                        )

                        convert_dashboard = gr.HTML("")
                        convert_status = gr.Textbox(
                            label="Status", lines=15, interactive=False,
                        )

                        convert_refresh_btn.click(
                            fn=lambda: gr.update(choices=discover_hf_model_dirs()),
                            outputs=convert_model_dir,
                        )

                        convert_btn.click(
                            fn=handle_convert_to_gguf,
                            inputs=[convert_model_dir, convert_out_type, convert_output_name],
                            outputs=[convert_dashboard, convert_status],
                        )

                    # --- Quantize GGUF sub-tab ---
                    with gr.Tab("Quantize GGUF"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=3, min_width=0):
                                quant_input_file = gr.Dropdown(
                                    choices=discover_gguf_files(),
                                    label="Input GGUF File",
                                    allow_custom_value=True,
                                    info="Select a GGUF file or paste a path",
                                )
                                with gr.Row():
                                    quant_refresh_btn = gr.Button(
                                        "Refresh", size="sm", scale=0, min_width=50,
                                    )

                            with gr.Column(scale=2, min_width=0):
                                quant_type_toggle = gr.Radio(
                                    choices=["Recommended", "All Types"],
                                    value="Recommended",
                                    label="Quantization Types",
                                )
                                quant_type_select = gr.Dropdown(
                                    choices=RECOMMENDED_QUANT_TYPES,
                                    value="Q4_K_M",
                                    label="Quantization Type",
                                    info="Q4_K_M is the most popular balance of size vs quality",
                                )
                                quant_output_name = gr.Textbox(
                                    label="Output Filename (optional)",
                                    placeholder="Auto-generated from input name + quant type",
                                    info="Saved to ./models/",
                                )

                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=0):
                                    quant_threads = gr.Slider(
                                        minimum=0, maximum=32, value=0, step=1,
                                        label="Threads",
                                        info="0 = use all available threads",
                                    )
                                with gr.Column(min_width=0):
                                    quant_allow_requantize = gr.Checkbox(
                                        value=False,
                                        label="Allow requantize",
                                        info="Allow re-quantizing already quantized tensors",
                                    )
                                    quant_leave_output = gr.Checkbox(
                                        value=False,
                                        label="Leave output tensor",
                                        info="Keep the output.weight tensor unquantized",
                                    )
                                    quant_pure = gr.Checkbox(
                                        value=False,
                                        label="Pure quantization",
                                        info="Disable k-quant mixtures, quantize all tensors to the same type",
                                    )

                        quant_btn = gr.Button(
                            "Quantize", variant="primary", size="lg",
                        )

                        quant_dashboard = gr.HTML("")
                        quant_status = gr.Textbox(
                            label="Status", lines=15, interactive=False,
                        )

                        # Wire quant type toggle
                        quant_type_toggle.change(
                            fn=lambda t: gr.update(
                                choices=RECOMMENDED_QUANT_TYPES if t == "Recommended" else ALL_QUANT_TYPES,
                                value="Q4_K_M" if t == "Recommended" else None,
                            ),
                            inputs=quant_type_toggle,
                            outputs=quant_type_select,
                        )

                        quant_refresh_btn.click(
                            fn=lambda: gr.update(choices=discover_gguf_files()),
                            outputs=quant_input_file,
                        )

                        quant_btn.click(
                            fn=handle_quantize_gguf,
                            inputs=[
                                quant_input_file, quant_type_select, quant_output_name,
                                quant_threads, quant_allow_requantize, quant_leave_output, quant_pure,
                            ],
                            outputs=[quant_dashboard, quant_status],
                        )

            # === About Tab ===
            with gr.Tab("About", id="about"):
                gr.Markdown(
                    """
                    ### Heretic Converter

                    A GUI for decensoring and de-slopping language models using
                    [Heretic](https://github.com/p-e-w/heretic), with built-in model
                    management and LAN transfer support.

                    **Companion app:** [MLX Model Converter](https://github.com/Brinven/MLX-Convertor) for
                    Apple Silicon conversion.

                    **Workflow:**
                    1. Decensor a model on your Windows PC (NVIDIA GPU)
                    2. Download the zip from the Models tab
                    3. Upload it to the MLX Converter on your Mac
                    4. Convert to MLX format for local inference

                    **Requirements:**
                    - Windows or Linux with NVIDIA GPU
                    - Python 3.10+
                    - PyTorch with CUDA support

                    **Built with:**
                    - [Gradio](https://www.gradio.app/) — UI framework
                    - [Heretic](https://github.com/p-e-w/heretic) — Decensoring engine
                    - [Optuna](https://optuna.org/) — Hyperparameter optimization

                    **License:** AGPL-3.0 (inherits from Heretic)

                    **Version:** 0.1.0
                    """
                )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=9300,
        inbrowser=False,
    )
