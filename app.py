"""Heretic Converter — A Gradio app for decensoring and de-slopping language models."""

from pathlib import Path

import gradio as gr

from src.decensor import run_decensor
from src.evaluator import evaluate_model, chat_with_model
from src.utils import list_saved_models, zip_model, import_model_zip

DEFAULT_OUTPUT_DIR = "./models"


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
        yield f"❌ HuggingFace login failed: {e}"
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
        yield f"❌ HuggingFace login failed: {e}"
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
        return f"❌ HuggingFace login failed: {e}"
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
        lines.append(f"• **{m['name']}** — {m['size']}\n  `{m['path']}`")
    return "\n\n".join(lines)


# --- Download/Upload handlers ---

def handle_download(model_name):
    """Handle the download button click."""
    if not model_name:
        return gr.update(), "❌ Please select a model to download."

    models = list_saved_models(DEFAULT_OUTPUT_DIR)
    match = next((m for m in models if m["name"] == model_name), None)
    if not match:
        return gr.update(), f"❌ Model '{model_name}' not found."

    result = zip_model(match["path"])
    if result["success"]:
        return result["zip_path"], f"✅ {result['message']}"
    else:
        return gr.update(), f"❌ {result['message']}"


def handle_upload(file):
    """Handle the upload/import of a model zip."""
    if file is None:
        return "❌ No file uploaded."

    file_path = file.name if hasattr(file, "name") else str(file)
    result = import_model_zip(file_path, DEFAULT_OUTPUT_DIR)
    if result["success"]:
        return f"✅ {result['message']}"
    else:
        return f"❌ {result['message']}"


# --- Build Gradio UI ---

def create_app():
    """Create and return the Gradio app."""
    with gr.Blocks(title="Heretic Converter") as app:
        gr.HTML(
            """
            <div class="main-header">
                <h1>🔓 Heretic Converter</h1>
                <p>Decensor and de-slop language models, then transfer to your Mac for MLX conversion</p>
            </div>
            """
        )

        with gr.Accordion("🔑 HuggingFace Token (required for gated models)", open=False):
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
            with gr.Tab("🔓 Decensor", id="decensor"):
                gr.Markdown("### Decensor a HuggingFace model")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "Browse models at [huggingface.co/models](https://huggingface.co/models)"
                        )
                        decensor_model_path = gr.Textbox(
                            label="HuggingFace Model Path",
                            placeholder="e.g., Qwen/Qwen3-4B-Instruct-2507",
                            info="Model identifier from huggingface.co",
                        )
                        decensor_output_name = gr.Textbox(
                            label="Output Name (optional)",
                            placeholder="Leave blank for auto-generated name",
                            info="Name for the output model directory",
                        )
                        decensor_mode = gr.Radio(
                            choices=["Decensor", "De-slop"],
                            value="Decensor",
                            label="Mode",
                            info="Decensor: remove safety alignment. De-slop: remove purple prose.",
                        )

                    with gr.Column(scale=1):
                        decensor_n_trials = gr.Slider(
                            minimum=10, maximum=500, value=200, step=10,
                            label="Optimization Trials",
                            info="More trials = better results but slower",
                        )
                        decensor_quantization = gr.Dropdown(
                            choices=["None", "8-bit (bitsandbytes)", "4-bit (bitsandbytes)"],
                            value="None",
                            label="Quantization",
                            info="4-bit reduces VRAM usage but may affect quality",
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

                decensor_btn = gr.Button(
                    "🚀 Start Decensoring", variant="primary", size="lg"
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
            with gr.Tab("📊 Evaluate", id="evaluate"):
                gr.Markdown("### Evaluate a decensored model")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            eval_model_path = gr.Dropdown(
                                choices=get_model_choices(),
                                label="Model",
                                allow_custom_value=True,
                                info="Select a decensored model or enter a HuggingFace path",
                                scale=4,
                            )
                            eval_refresh_btn = gr.Button(
                                "🔄", size="sm", scale=0, min_width=50
                            )
                        eval_prompt = gr.Textbox(
                            label="Test Prompt (optional)",
                            placeholder="Enter a prompt to test, or leave blank for automated evaluation",
                            lines=3,
                        )

                    with gr.Column(scale=1):
                        eval_max_tokens = gr.Slider(
                            minimum=50, maximum=500, value=100, step=10,
                            label="Max Tokens",
                        )
                        eval_quantization = gr.Dropdown(
                            choices=["None", "8-bit (bitsandbytes)", "4-bit (bitsandbytes)"],
                            value="None",
                            label="Quantization",
                        )

                with gr.Row():
                    eval_btn = gr.Button(
                        "📊 Run Evaluation", variant="primary", size="lg"
                    )
                    chat_btn = gr.Button("💬 Test Chat", size="lg")

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
            with gr.Tab("📂 Models", id="models"):
                gr.Markdown("### Saved Models")
                models_display = gr.Markdown(
                    "Click Refresh to see saved models."
                )
                with gr.Row():
                    models_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    model_selector = gr.Dropdown(
                        choices=get_model_names(),
                        label="Select Model",
                        scale=3,
                    )
                models_refresh_btn.click(
                    fn=lambda: (
                        refresh_model_list(),
                        gr.update(choices=get_model_names()),
                    ),
                    outputs=[models_display, model_selector],
                )

                gr.Markdown("### Download Model")
                gr.Markdown(
                    "Download a model as a zip file for transfer to your Mac."
                )
                download_btn = gr.Button(
                    "📥 Download as Zip", variant="primary"
                )
                download_status = gr.Textbox(
                    label="Status", lines=2, interactive=False
                )
                download_file = gr.File(
                    label="Download", interactive=False
                )
                download_btn.click(
                    fn=handle_download,
                    inputs=model_selector,
                    outputs=[download_file, download_status],
                )

                gr.Markdown("### Import Model")
                gr.Markdown(
                    "Upload a model zip file to import into the models directory."
                )
                upload_file = gr.File(
                    label="Upload Model Zip", file_types=[".zip"]
                )
                import_status = gr.Textbox(
                    label="Import Status", lines=2, interactive=False
                )
                upload_file.change(
                    fn=handle_upload,
                    inputs=upload_file,
                    outputs=import_status,
                )

            # === About Tab ===
            with gr.Tab("ℹ️ About", id="about"):
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
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 0.5em; }
        .main-header h1 { margin-bottom: 0.1em; }
        .main-header p { color: #666; font-size: 0.95em; }
        """,
    )
