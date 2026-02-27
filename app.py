"""Heretic Converter — A Gradio app for decensoring and de-slopping language models."""

from pathlib import Path

import gradio as gr
from gradio.themes import Base
from gradio.themes.utils import colors, sizes, fonts

from src.decensor import run_decensor
from src.evaluator import evaluate_model, chat_with_model
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
    model_path, output_name, mode, n_trials, quantization, batch_size, max_response_length, hf_token
):
    """Handle the decensor button click."""
    try:
        _login_hf(hf_token)
    except Exception as e:
        yield f"HuggingFace login failed: {e}"
        return
    yield from run_decensor(
        model_path=model_path,
        output_name=output_name,
        mode=mode,
        n_trials=int(n_trials),
        quantization=quantization,
        batch_size=int(batch_size),
        max_response_length=int(max_response_length),
        output_dir=DEFAULT_OUTPUT_DIR,
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


# --- Build Gradio UI ---

def create_app():
    """Create and return the Gradio app."""
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
                        decensor_output_name = gr.Textbox(
                            label="Output Name (optional)",
                            placeholder="Leave blank for auto-generated name",
                            info="Name for the output model directory",
                        )

                    with gr.Column(scale=2, min_width=0):
                        decensor_n_trials = gr.Slider(
                            minimum=10, maximum=500, value=200, step=10,
                            label="Optimization Trials",
                            info="More trials = better results but slower",
                        )
                        decensor_quantization = gr.Dropdown(
                            choices=["None", "4-bit (bitsandbytes)"],
                            value="None",
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
                        hf_token,
                    ],
                    outputs=decensor_status,
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
