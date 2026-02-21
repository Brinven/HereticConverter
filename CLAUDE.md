# Heretic Converter — Decensor & De-slop GUI

## Project Overview
A Gradio web app wrapping the [Heretic](https://github.com/p-e-w/heretic) decensoring/de-slopping tool with a polished GUI. Companion app to the [MLX Model Converter](../MLX%20Convertor/) — decensor on a Windows PC with an NVIDIA GPU, then transfer the model over LAN to a Mac for MLX conversion.

## Target Platform
- **Primary:** Windows with NVIDIA GPU (CUDA)
- **Also works on:** Linux with NVIDIA GPU
- **Note:** Heretic requires PyTorch with CUDA. Apple Silicon is NOT supported.

## Technical Stack
- **Framework:** Gradio (web-based UI, matches the MLX Converter)
- **Backend:** heretic-llm for decensoring/de-slopping and evaluation
- **Language:** Python 3.10+
- **Key deps:** PyTorch (CUDA), transformers, accelerate, optuna, peft, bitsandbytes

## Visual & UX Spec

### Must match the MLX Converter exactly:
- **Theme:** `gr.themes.Soft()`
- **CSS:** Same custom CSS block
- **Header:** Centered `<div class="main-header">` with emoji + title + subtitle
- **Tabs:** Emoji-prefixed tab labels (`"🔓 Decensor"`, `"📊 Evaluate"`, `"📂 Models"`, `"ℹ️ About"`)
- **Layout:** `gr.Row()` with `gr.Column(scale=2)` for main inputs, `gr.Column(scale=1)` for settings/params
- **Buttons:** Primary action buttons use `variant="primary", size="lg"` with emoji prefix
- **Status output:** `gr.Textbox(label="Status", lines=5, interactive=False, show_copy_button=True)`
- **Success/error prefixes:** `"✅ ..."` and `"❌ ..."` in status messages
- **Server binding:** `app.launch(server_name="0.0.0.0", inbrowser=True)` for LAN access

### CSS (copy exactly from MLX app):
```css
.main-header { text-align: center; margin-bottom: 0.5em; }
.main-header h1 { margin-bottom: 0.1em; }
.main-header p { color: #666; font-size: 0.95em; }
```

### Header HTML:
```html
<div class="main-header">
    <h1>🔓 Heretic Converter</h1>
    <p>Decensor and de-slop language models, then transfer to your Mac for MLX conversion</p>
</div>
```

## Tab-by-Tab UI Spec

### Tab 1: 🔓 Decensor
**Purpose:** Run Heretic's abliteration optimization on a HuggingFace model.

**Layout:**
```
gr.Markdown("### Decensor a HuggingFace model")

gr.Row():
    gr.Column(scale=2):  # Main inputs
        model_path      — Textbox, label="HuggingFace Model Path"
                           placeholder="e.g., Qwen/Qwen3-4B-Instruct-2507"
                           info="Model identifier from huggingface.co"
        output_name     — Textbox, label="Output Name (optional)"
                           placeholder="Leave blank for auto-generated name"
        mode            — Radio, choices=["Decensor", "De-slop"], value="Decensor"
                           label="Mode"
                           info="Decensor: remove safety alignment. De-slop: remove purple prose."

    gr.Column(scale=1):  # Parameters
        n_trials        — Slider, min=10, max=500, value=200, step=10
                           label="Optimization Trials"
                           info="More trials = better results but slower"
        quantization    — Dropdown, choices=["None", "4-bit (bitsandbytes)"], value="None"
                           label="Quantization"
                           info="4-bit reduces VRAM usage but may affect quality"
        batch_size      — Slider, min=0, max=128, value=0, step=1
                           label="Batch Size"
                           info="0 = auto-detect optimal batch size"
        max_response_length — Slider, min=50, max=500, value=100, step=10
                           label="Max Response Length"
                           info="Tokens generated per evaluation response"

gr.Button("🚀 Start Decensoring", variant="primary", size="lg")
gr.Textbox(label="Status", lines=8, interactive=False, show_copy_button=True)
```

**Handler:** `handle_decensor(model_path, output_name, mode, n_trials, quantization, batch_size, max_response_length)`
- Validates inputs
- Creates Settings programmatically (bypassing CLI)
- Runs the decensoring pipeline
- Reports progress via status updates
- On success: saves model to `./models/{output_name}/` and reports stats

### Tab 2: 📊 Evaluate
**Purpose:** Evaluate an existing decensored model's refusal rate and KL divergence.

**Layout:**
```
gr.Markdown("### Evaluate a decensored model")

gr.Row():
    gr.Column(scale=2):
        gr.Row():
            model_path  — Dropdown + allow_custom_value
                           label="Model"
                           info="Select a decensored model or enter a HuggingFace path"
            refresh_btn — Button("🔄", size="sm")
        prompt          — Textbox, label="Test Prompt (optional)"
                           placeholder="Enter a prompt to test, or leave blank for automated evaluation"
                           lines=3

    gr.Column(scale=1):
        max_tokens      — Slider, min=50, max=500, value=100, step=10
                           label="Max Tokens"
        quantization    — Dropdown, choices=["None", "4-bit (bitsandbytes)"], value="None"
                           label="Quantization"

gr.Row():
    gr.Button("📊 Run Evaluation", variant="primary", size="lg")
    gr.Button("💬 Test Chat", size="lg")

gr.Textbox(label="Results", lines=10, interactive=False, show_copy_button=True)
```

**Handlers:**
- `handle_evaluate(model_path, quantization)` — runs Evaluator, reports refusal count + KL divergence
- `handle_chat(model_path, prompt, max_tokens, quantization)` — single prompt generation for manual testing

### Tab 3: 📂 Models
**Purpose:** Browse saved models. Download as zip for LAN transfer. Upload/import zips.

**Layout:**
```
gr.Markdown("### Saved Models")

models_display  — Markdown, shows list of models with metadata
gr.Row():
    refresh_btn     — Button("🔄 Refresh", size="sm")
    model_selector  — Dropdown, label="Select Model", allow_custom_value=False

gr.Markdown("### Download Model")
download_btn    — Button("📥 Download as Zip", variant="primary")
download_file   — gr.File(label="Download", interactive=False)
                   # This gr.File component serves the zip for browser download

gr.Markdown("### Import Model")
upload_file     — gr.File(label="Upload Model Zip", file_types=[".zip"])
import_status   — Textbox(label="Import Status", lines=3, interactive=False)
```

**Handlers:**
- `refresh_model_list()` — same pattern as MLX app, scan `./models/` for dirs with config.json or .safetensors
- `handle_download(model_name)` — zip the model directory, return path for `gr.File`
- `handle_upload(file)` — extract uploaded zip to `./models/`, validate contents

### Tab 4: ℹ️ About
**Purpose:** Version, links, license notice.

```
gr.Markdown("""
### Heretic Converter

A GUI for decensoring and de-slopping language models using
[Heretic](https://github.com/p-e-w/heretic), with built-in model
management and LAN transfer support.

**Companion app:** [MLX Model Converter](https://github.com/...) for
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
""")
```

## LAN Transfer Feature

Both apps (this one and MLX Converter) are bound to `0.0.0.0`, so they're accessible across the LAN. The transfer workflow uses Gradio's built-in `gr.File` component:

### Download (this app — Windows side):
1. User selects a model in the Models tab
2. Clicks "Download as Zip"
3. Handler zips the model directory (config.json + safetensors + tokenizer files)
4. Returns the zip path to a `gr.File(interactive=False)` component
5. User (or Mac browser) clicks the download link

### Upload/Import (MLX Converter — Mac side):
1. User opens MLX Converter's Models tab
2. Uploads the zip via `gr.File(file_types=[".zip"])`
3. Handler extracts to `./models/` directory
4. Model appears in the model list, ready for MLX conversion

### Implementation notes:
- Use `shutil.make_archive()` for zipping, `zipfile.ZipFile` for extraction
- Validate zip contents before extraction (must contain config.json or .safetensors)
- Show file size in status messages
- Clean up temp zip files after download completes

## Heretic Integration Guide

### Package info
- **PyPI package:** `heretic-llm` (install as `pip install heretic-llm`)
- **Import as:** `heretic` (not `heretic_llm`)
- **Version:** 1.2.0+
- **License:** AGPL-3.0-or-later

### Key classes (import from submodules):
```python
from heretic.config import Settings, DatasetSpecification, QuantizationMethod
from heretic.model import Model, AbliterationParameters
from heretic.evaluator import Evaluator
from heretic.utils import load_prompts, empty_cache, Prompt
```

### Bypassing the CLI
Heretic's `Settings` class uses `pydantic_settings.BaseSettings` which tries to parse `sys.argv` by default. To bypass:

**Option A — Patch sys.argv before constructing Settings:**
```python
import sys
original_argv = sys.argv
sys.argv = [sys.argv[0]]  # Strip CLI args
try:
    settings = Settings(model="org/model-name", n_trials=200)
finally:
    sys.argv = original_argv
```

**Option B — Construct Settings with `_cli_parse_args=False` (if supported by pydantic-settings version):**
```python
settings = Settings(model="org/model-name", _cli_parse_args=False)
```

Test both approaches and use whichever works with the installed pydantic-settings version.

### Core algorithm flow (from `main.py:run()`):
1. **Load model:** `model = Model(settings)` — loads HF model, wraps with LoRA adapters
2. **Load prompts:** `good_prompts = load_prompts(settings, settings.good_prompts)` (and bad_prompts)
3. **Auto-detect batch size:** Binary search loop if `settings.batch_size == 0`
4. **Detect response prefix:** `model.get_responses_batched(...)` + `commonprefix()`
5. **Compute refusal directions:** Get residuals for good/bad prompts, calculate `F.normalize(bad_means - good_means)`
6. **Create evaluator:** `evaluator = Evaluator(settings, model)` — loads eval prompts, gets baseline logprobs
7. **Optuna optimization loop:** `n_trials` iterations, each trial:
   - Suggest direction_index + AbliterationParameters for each component
   - `model.reset_model()` + `model.abliterate(refusal_directions, direction_index, params)`
   - `evaluator.get_score()` returns `((kld_score, refusals_score), kl_divergence, refusals)`
   - Objective minimizes refusal count + KL divergence
8. **Apply best parameters:** Re-abliterate with best trial's params
9. **Save:** `model.get_merged_model().save_pretrained(path)` + `model.tokenizer.save_pretrained(path)`

### De-slop mode
When mode is "De-slop", load `config.noslop.toml` settings which use different datasets (`llm-aes/writing-prompts` with slop-inducing vs slop-suppressing prefixes) and different refusal markers (literary cliche words). Implement by overriding the relevant Settings fields.

### AbliterationParameters fields:
- `max_weight` — Peak ablation strength
- `max_weight_position` — Layer position of peak
- `min_weight` — Minimum ablation strength
- `min_weight_distance` — Distance from peak to minimum

### Model components abliterated:
- `attn.o_proj` — Attention output projection
- `mlp.down_proj` — MLP down projection

Each gets its own set of AbliterationParameters, optimized independently.

## Architecture

### File structure:
```
HereticConverter/
├── CLAUDE.md                    # This file
├── README.md                    # User-facing documentation
├── requirements.txt             # Python dependencies (not PyTorch)
├── .gitignore
├── app.py                       # Main Gradio application
├── src/
│   ├── __init__.py
│   ├── decensor.py             # Decensoring pipeline (wraps Heretic)
│   ├── evaluator.py            # Model evaluation logic
│   └── utils.py                # Helpers (model listing, zip/unzip, formatting)
├── models/                      # Default output directory
└── configs/
    ├── decensor_defaults.json  # Default decensor settings
    └── deslop_defaults.json    # Default de-slop settings (overrides for noslop mode)
```

### Module responsibilities:

**`src/decensor.py`** — Core decensoring pipeline
- `run_decensor(model_path, output_name, mode, n_trials, quantization, batch_size, max_response_length, output_dir)` → dict
- Creates Heretic `Settings`, `Model`, runs the optimization loop, saves result
- Returns `{"success": bool, "message": str, "output_path": str, "stats": dict}`
- Stats include: refusal_count, kl_divergence, n_trials_completed, duration

**`src/evaluator.py`** — Evaluation wrapper
- `evaluate_model(model_path, quantization)` → dict with refusal count, KL divergence, responses
- `chat_with_model(model_path, prompt, max_tokens, quantization)` → dict with response text

**`src/utils.py`** — Shared helpers (match MLX app patterns)
- `get_model_size(path)` → int (bytes)
- `format_size(size_bytes)` → str ("1.5 GB")
- `list_saved_models(output_dir)` → list of dicts
- `zip_model(model_path)` → str (path to temp zip file)
- `import_model_zip(zip_path, output_dir)` → dict with success/message
- `validate_model_dir(path)` → bool

### Handler pattern (match MLX app exactly):
```python
def handle_decensor(model_path, output_name, mode, ...):
    """Handle the decensor button click."""
    result = run_decensor(model_path=model_path, ...)
    if result["success"]:
        return f"✅ {result['message']}"
    else:
        return f"❌ {result['message']}"
```

### Error message conventions:
- Model not found: `"Model '{path}' was not found on HuggingFace. Please check the model path."`
- No GPU: `"No CUDA GPU detected. Heretic requires an NVIDIA GPU with CUDA support."`
- Out of VRAM: `"Out of GPU memory. Try enabling 4-bit quantization or using a smaller model."`
- Network error: `"Network error. Please check your internet connection and try again."`
- Invalid zip: `"Invalid model zip. Archive must contain config.json or .safetensors files."`

### Docstring style (match MLX app):
```python
def function_name(param: type) -> return_type:
    """One-line description.

    Args:
        param: Description.

    Returns:
        Description.
    """
```

## Dependencies

### requirements.txt (do NOT include PyTorch — installed separately per CUDA version):
```
gradio>=4.0
heretic-llm>=1.2.0
accelerate~=1.10
bitsandbytes~=0.45
datasets~=4.0
huggingface-hub~=0.34
optuna~=4.5
peft~=0.14
pydantic-settings~=2.10
transformers~=4.57
rich~=14.1
```

### PyTorch installation (documented in README, not in requirements.txt):
```bash
# Install PyTorch with CUDA support first:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# Then install the app:
pip install -r requirements.txt
```

## Implementation Phases

### Phase 1: MVP
1. **Setup** — Project structure, requirements, README
2. **Decensor module** (`src/decensor.py`) — Wrap Heretic's core pipeline
3. **Evaluator module** (`src/evaluator.py`) — Model evaluation and chat
4. **Utils module** (`src/utils.py`) — Model listing, zip/unzip, size formatting
5. **Main app** (`app.py`) — Gradio UI with all 4 tabs
6. **Test** — Run decensoring on a small model (e.g., Qwen3-0.6B)

### Phase 2: Polish
1. UI refinements, tooltips, better progress reporting
2. De-slop mode with noslop config
3. Model metadata display (date, settings used, stats)
4. Configs directory with default/noslop JSON files

### Phase 3: Distribution
1. README with screenshots
2. Clean up, LICENSE file (AGPL-3.0)
3. GitHub release

## Testing Checklist
- [ ] Decensoring with valid HuggingFace model path
- [ ] Decensoring with invalid model path
- [ ] Decensoring with 4-bit quantization enabled
- [ ] Decensoring with no GPU available (graceful error)
- [ ] Evaluation of a decensored model
- [ ] Chat with a decensored model
- [ ] Model list refresh
- [ ] Download model as zip
- [ ] Import model from zip
- [ ] Invalid zip upload (graceful error)
- [ ] UI responsive during long operations
- [ ] LAN access from another machine
- [ ] De-slop mode works

## Default Values
- **Mode:** Decensor
- **Optimization trials:** 200
- **Quantization:** None
- **Batch size:** 0 (auto)
- **Max response length:** 100
- **Output directory:** `./models/`

## References
- Heretic: https://github.com/p-e-w/heretic
- Heretic PyPI: https://pypi.org/project/heretic-llm/
- Gradio: https://www.gradio.app/docs
- Optuna: https://optuna.readthedocs.io/
- MLX Converter (companion app): ../MLX%20Convertor/

## Notes for Claude Code
- Start with MVP — get basic decensoring working first
- Match the MLX Converter's visual style exactly (theme, CSS, layout patterns)
- Use threading to prevent UI freezing during decensoring (it's very long-running)
- The Heretic pipeline is CPU/GPU intensive and takes 30-60+ minutes — provide progress updates
- Test the Settings bypass approach early (sys.argv patching)
- The `run()` function in Heretic's main.py is monolithic and interactive — we must replicate its logic, not call it directly
- Keep model zip files reasonable — large models may be 5-15 GB
