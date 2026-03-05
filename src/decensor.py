"""Core decensoring pipeline wrapping Heretic's abliteration engine."""

import io
import os
import sys
import time
import warnings
from contextlib import contextmanager
from os.path import commonprefix
from pathlib import Path
from typing import Generator

from src.utils import format_duration, format_size, get_model_size

# De-slop refusal markers — literary cliches and purple prose indicators.
# When mode is "De-slop", these replace the default refusal markers.
DESLOP_REFUSAL_MARKERS = [
    "tapestry",
    "testament",
    "beacon",
    "crucible",
    "symphony",
    "echoes of",
    "remnants of",
    "dance of",
    "veil of",
    "labyrinth",
    "enigma",
    "paradox",
    "catalyst",
    "metamorphosis",
    "alchemy",
    "whispers of",
    "shadows of",
    "fabric of",
    "realm of",
    "tendrils",
    "gossamer",
    "luminescent",
    "resplendent",
    "effervescent",
    "iridescent",
    "ethereal",
    "serendipitous",
    "juxtaposition",
    "paradigm",
    "quintessential",
]

# De-slop writing prompt prefixes
DESLOP_GOOD_PREFIX = (
    "Write in a clear, direct, and natural style. "
    "Avoid flowery language, purple prose, and literary cliches. "
)
DESLOP_BAD_PREFIX = (
    "Write in a highly literary, ornate style with rich metaphors, "
    "poetic imagery, and elaborate descriptions. "
)


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

def _kl_color(kl: float) -> str:
    """Return CSS color for a KL divergence value."""
    if kl < 0.8:
        return "#10b981"
    elif kl <= 1.5:
        return "#eab308"
    return "#ef4444"


def _health_label(kl: float) -> tuple[str, str]:
    """Return (label, css_color) for model health based on KL divergence."""
    if kl < 0.8:
        return "Healthy", "#10b981"
    elif kl <= 1.5:
        return "Moderate Drift", "#eab308"
    return "Potentially Damaged", "#ef4444"


def _make_dashboard_html(
    current_kl: float | None = None,
    best_kl: float | None = None,
    best_refusals: int | None = None,
    best_trial: int = 0,
    improvements: int = 0,
    n_bad: int | None = None,
    kl_ceiling: float = 1.0,
    refusal_target: int = 5,
    kl_exceeded_count: int = 0,
) -> str:
    """Build the real-time KL / results dashboard HTML."""
    if current_kl is None:
        return ""

    bk = best_kl if best_kl is not None else current_kl
    health_text, health_color = _health_label(bk)
    cur_color = _kl_color(current_kl)
    best_color = _kl_color(bk)

    ref_str = str(best_refusals) if best_refusals is not None else "---"
    if n_bad and best_refusals is not None:
        ref_str = f"{best_refusals}/{n_bad}"

    if best_refusals is not None:
        ref_color = (
            "#10b981" if best_refusals <= refusal_target
            else "#eab308" if best_refusals <= refusal_target * 2
            else "#ef4444"
        )
    else:
        ref_color = "#71717a"

    best_kl_str = f"{bk:.4f}" if best_kl is not None else "---"

    exceeded_note = ""
    if kl_exceeded_count > 0:
        exceeded_note = (
            f'<div style="color:#ef4444;font-size:0.7em;">'
            f'{kl_exceeded_count} trial{"s" if kl_exceeded_count != 1 else ""} exceeded ceiling</div>'
        )

    return (
        '<div style="display:flex;gap:1rem;padding:0.75rem;background:#18181b;'
        'border:1px solid #27272a;border-radius:8px;font-family:\'Geist Mono\',monospace;">'
        # Current KL
        '<div style="flex:1;text-align:center;">'
        '<div style="color:#a1a1aa;font-size:0.75em;">Current KL</div>'
        f'<div style="color:{cur_color};font-size:1.3em;font-weight:bold;">{current_kl:.4f}</div>'
        '</div>'
        # Best KL
        '<div style="flex:1;text-align:center;">'
        '<div style="color:#a1a1aa;font-size:0.75em;">Best KL</div>'
        f'<div style="color:{best_color};font-size:1.3em;font-weight:bold;">{best_kl_str}</div>'
        f'<div style="color:#71717a;font-size:0.7em;">Ceiling: {kl_ceiling}</div>'
        f'{exceeded_note}'
        '</div>'
        # Refusals
        '<div style="flex:1;text-align:center;">'
        '<div style="color:#a1a1aa;font-size:0.75em;">Refusals</div>'
        f'<div style="color:{ref_color};font-size:1.3em;font-weight:bold;">{ref_str}</div>'
        f'<div style="color:#71717a;font-size:0.7em;">Target: {refusal_target}</div>'
        '</div>'
        # Model Health
        '<div style="flex:1;text-align:center;">'
        '<div style="color:#a1a1aa;font-size:0.75em;">Model Health</div>'
        f'<div style="color:{health_color};font-size:1.1em;font-weight:bold;">{health_text}</div>'
        f'<div style="color:#71717a;font-size:0.7em;">Trial #{best_trial} | {improvements} impr.</div>'
        '</div>'
        '</div>'
    )


def _is_better(refs, kl, best_refs, best_kl, target):
    """Compare trial results, prioritising refusal target then KL."""
    refs_met = refs <= target
    best_met = best_refs <= target
    if refs_met and best_met:
        return kl < best_kl
    elif refs_met:
        return True
    elif best_met:
        return False
    return refs < best_refs or (refs == best_refs and kl < best_kl)


def _trial_sort_key(t, target):
    """Sort key for selecting the best trial (refusal-target-aware)."""
    r = t.user_attrs["refusals"]
    k = t.user_attrs["kl_divergence"]
    meets = 0 if r <= target else 1
    # Trials that meet the refusal target sort first; among them, prefer lower KL.
    # Trials that don't meet the target sort second; among them, prefer fewer refusals then lower KL.
    return (meets, k if meets == 0 else r, k if meets == 1 else r)


# ---------------------------------------------------------------------------
# Transformers tokenizer compatibility patch
# ---------------------------------------------------------------------------

@contextmanager
def _capture_heretic_output():
    """Capture heretic's rich console output so we can surface per-dtype errors.

    Heretic's Model.__init__ prints detailed per-dtype errors via a `print`
    imported from heretic.utils (bound to a rich Console).  Since model.py
    imports `print` at module level, we must patch the reference in both
    heretic.utils AND heretic.model to intercept it.
    """
    captured = io.StringIO()
    try:
        from rich.console import Console
        import heretic.utils as _hu
        import heretic.model as _hm
        capture_print = Console(
            file=captured, force_terminal=False, no_color=True, highlight=False,
        ).print
        orig_utils_print = _hu.print
        orig_model_print = _hm.print
        _hu.print = capture_print
        _hm.print = capture_print
        try:
            yield captured
        finally:
            _hu.print = orig_utils_print
            _hm.print = orig_model_print
    except (ImportError, AttributeError):
        yield captured


@contextmanager
def _patch_hybrid_lora():
    """Patch Model._apply_lora to discover abliterable modules from ALL layers.

    Heretic's _apply_lora() only examines layer 0 to decide which modules
    get LoRA adapters.  Hybrid architectures (e.g. Qwen3.5 with alternating
    DeltaNet and Attention layers) have different modules per layer type.
    Without this patch, modules only present in non-zero layers (like
    attn.o_proj in attention layers) won't get LoRA, and abliterate() will
    crash when it tries to write LoRA weights to them.

    This patches _apply_lora to scan every layer for target modules so that
    LoRA covers the full union of components.
    """
    from heretic.model import Model
    original_apply_lora = Model._apply_lora

    def _patched_apply_lora(self):
        from typing import cast as _cast
        from torch.nn import Module as _Module
        from peft import LoraConfig, PeftModel, get_peft_model
        from heretic.config import RowNormalization

        assert isinstance(self.model, __import__("transformers").PreTrainedModel)

        # Collect target module IDs from ALL layers (not just layer 0)
        target_ids = set()
        for i in range(len(self.get_layers())):
            for mods in self.get_layer_modules(i).values():
                for mod in mods:
                    if isinstance(mod, _Module):
                        target_ids.add(id(mod))

        target_modules = list({
            name.split(".")[-1]
            for name, mod in self.model.named_modules()
            if id(mod) in target_ids
        })

        # Same as original — only the target discovery above is changed
        if self.settings.row_normalization != RowNormalization.FULL:
            lora_rank = 1
        else:
            lora_rank = self.settings.full_normalization_lora_rank

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = _cast(PeftModel, get_peft_model(self.model, self.peft_config))

        from heretic.utils import print as _hprint
        _hprint(f"* LoRA adapters initialized (targets: {', '.join(target_modules)})")

    Model._apply_lora = _patched_apply_lora
    try:
        yield
    finally:
        Model._apply_lora = original_apply_lora


@contextmanager
def _patch_slow_tokenizer():
    """Wrap slow tokenizer creation in try-except for transformers ~4.57.

    transformers' _from_pretrained (tokenization_utils_base.py:2151) tries to
    create a slow tokenizer when tokenizer.json is missing.  If the slow
    tokenizer's vocab files aren't available either, it crashes with a
    TypeError instead of falling back gracefully.  Newer transformers
    versions wrap this in try-except; we backport that behaviour here.
    """
    try:
        import transformers.tokenization_utils_base as _tub
    except ImportError:
        yield
        return

    _orig_func = _tub.PreTrainedTokenizerBase._from_pretrained.__func__

    def _safe_from_pretrained(cls, resolved_vocab_files, *args, **kwargs):
        # Force from_slow=False when tokenizer.json IS present — skip slow
        # tokenizer entirely to avoid crashes from missing vocab files.
        has_tokenizer_file = resolved_vocab_files.get("tokenizer_file") is not None
        if has_tokenizer_file:
            kwargs["from_slow"] = False
        try:
            return _orig_func(cls, resolved_vocab_files, *args, **kwargs)
        except (TypeError, AttributeError) as exc:
            if "NoneType" not in str(exc):
                raise
            # Slow tokenizer failed — retry forcing from_slow=False
            kwargs["from_slow"] = False
            return _orig_func(cls, resolved_vocab_files, *args, **kwargs)

    _tub.PreTrainedTokenizerBase._from_pretrained = classmethod(_safe_from_pretrained)
    try:
        yield
    finally:
        _tub.PreTrainedTokenizerBase._from_pretrained = classmethod(_orig_func)


# ---------------------------------------------------------------------------
# Settings construction
# ---------------------------------------------------------------------------

def _make_settings(
    model_path: str,
    n_trials: int,
    quantization: str,
    batch_size: int,
    max_response_length: int,
    mode: str,
):
    """Construct Heretic Settings, bypassing CLI argument parsing."""
    from heretic.config import Settings, QuantizationMethod

    quant = None
    if quantization == "4-bit (bitsandbytes)":
        quant = QuantizationMethod.BNB_4BIT

    kwargs = {
        "model": model_path,
        "n_trials": n_trials,
        "max_response_length": max_response_length,
        # Always set a non-zero batch_size — Heretic's batchify() crashes with 0.
        # If user selected auto (0), start with 32; we'll auto-tune down on OOM later.
        "batch_size": batch_size if batch_size > 0 else 32,
        # Many models (Qwen, Phi, etc.) ship custom tokenizer code that
        # AutoTokenizer can only use when trust_remote_code is True.
        # Without it, transformers falls back to a generic class
        # (e.g. RobertaTokenizerFast) that can't find the right vocab files.
        "trust_remote_code": True,
    }

    if quant is not None:
        kwargs["quantization"] = quant

    # De-slop mode overrides
    if mode == "De-slop":
        kwargs["good_prompts"] = "llm-aes/writing-prompts"
        kwargs["bad_prompts"] = "llm-aes/writing-prompts"
        kwargs["good_system_prompt"] = DESLOP_GOOD_PREFIX
        kwargs["bad_system_prompt"] = DESLOP_BAD_PREFIX
        kwargs["refusal_markers"] = DESLOP_REFUSAL_MARKERS

    # Bypass sys.argv parsing — Settings uses pydantic-settings which reads CLI args
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        settings = Settings(**kwargs)
    finally:
        sys.argv = original_argv

    return settings


# ---------------------------------------------------------------------------
# Main decensoring pipeline
# ---------------------------------------------------------------------------

def run_decensor(
    model_path: str,
    output_name: str,
    mode: str,
    n_trials: int,
    quantization: str,
    batch_size: int,
    max_response_length: int,
    output_dir: str = "./models",
    kl_ceiling: float = 1.0,
    refusal_target: int = 5,
    target_output_layers_only: bool = True,
    layer_range_min: int = 0,
    layer_range_max: int = 0,
) -> Generator[tuple[str, str], None, None]:
    """Run the decensoring pipeline, yielding (dashboard_html, status_text) tuples.

    Each yield pushes a status update to the Gradio output components in real time.
    """
    start_time = time.time()
    model = None

    try:
        import torch
        import torch.nn.functional as F

        # --- Phase 1/8: Validate inputs ---
        yield ("", "Phase 1/8: Validating inputs...")

        if not model_path or not model_path.strip():
            yield ("", "Model path cannot be empty.")
            return

        model_path = model_path.strip()

        # Reject GGUF models — abliteration needs full-precision safetensors
        if "gguf" in model_path.lower() or model_path.lower().endswith(".gguf"):
            yield ("", (
                "GGUF models cannot be used for decensoring.\n\n"
                "Heretic's abliteration requires full-precision safetensors weights to\n"
                "access individual transformer layers and apply LoRA modifications.\n"
                "GGUF is a quantized single-file format that doesn't support this.\n\n"
                "Look for the base model without '-GGUF' in the name.\n"
                "For example, instead of:\n"
                "  mradermacher/SomeModel-GGUF\n"
                "Use the original:\n"
                "  original-author/SomeModel"
            ))
            return

        if not output_name or not output_name.strip():
            model_short = model_path.split("/")[-1]
            suffix = "-decensored" if mode == "Decensor" else "-deslopped"
            output_name = f"{model_short}{suffix}"
        else:
            output_name = output_name.strip()

        output_path = Path(output_dir) / output_name
        if output_path.exists():
            # Auto-increment version: name-v2, name-v3, etc.
            version = 2
            while (Path(output_dir) / f"{output_name}-v{version}").exists():
                version += 1
            output_name = f"{output_name}-v{version}"
            output_path = Path(output_dir) / output_name
            yield ("", f"Output already exists, using: {output_name}")

        cuda_available, cuda_info = _check_cuda()
        if not cuda_available:
            yield ("", f"No CUDA GPU detected. Heretic requires an NVIDIA GPU with CUDA support.\nDetails: {cuda_info}")
            return

        # Match Heretic's run() setup
        torch.set_grad_enabled(False)
        torch._dynamo.config.cache_size_limit = 64

        import transformers
        transformers.logging.set_verbosity_error()

        yield ("", f"Phase 1/8: Validated. GPU: {cuda_info}")

        # --- Phase 2/8: Construct Settings ---
        yield ("", "Phase 2/8: Configuring settings...")

        try:
            settings = _make_settings(
                model_path=model_path,
                n_trials=n_trials,
                quantization=quantization,
                batch_size=batch_size,
                max_response_length=max_response_length,
                mode=mode,
            )
        except Exception as e:
            yield ("", f"Failed to create settings: {e}")
            return

        layer_info = "output layers only" if target_output_layers_only else "all layers"
        if not target_output_layers_only and (layer_range_min > 0 or layer_range_max > 0):
            layer_info = f"layers {layer_range_min}-{layer_range_max}"

        yield ("", (
            f"Phase 2/8: Settings configured — {mode} mode, {n_trials} trials\n"
            f"  KL ceiling: {kl_ceiling} | Refusal target: {refusal_target} | Layers: {layer_info}"
        ))

        # --- Phase 3/8: Load model ---
        yield ("", f"Phase 3/8: Loading model '{model_path}'...\nThis may take 30-60 seconds and will download the model if not cached.")

        from heretic.model import Model, AbliterationParameters

        try:
            with _capture_heretic_output() as captured, _patch_slow_tokenizer(), _patch_hybrid_lora():
                model = Model(settings)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # Include heretic's per-dtype console output for diagnostics
            heretic_log = captured.getvalue().strip()
            detail = f"Failed to load model: {e}\n\nTraceback:\n{tb}"
            if heretic_log:
                detail += f"\n\nHeretic log (per-dtype attempts):\n{heretic_log}"
            yield ("", detail)
            return

        elapsed = format_duration(time.time() - start_time)
        yield ("", f"Phase 3/8: Model loaded ({elapsed})")

        # --- Phase 4/8: Load prompt datasets ---
        yield ("", "Phase 4/8: Loading prompt datasets...")

        from heretic.utils import load_prompts, empty_cache

        try:
            good_prompts = load_prompts(settings, settings.good_prompts)
            bad_prompts = load_prompts(settings, settings.bad_prompts)
        except Exception as e:
            yield ("", f"Failed to load prompt datasets: {e}")
            return

        yield ("", f"Phase 4/8: Loaded {len(good_prompts)} good + {len(bad_prompts)} bad prompts")

        # --- Phase 5/8: Detect prefix + compute refusal directions ---
        yield ("", f"Phase 5/8: Computing refusal directions (batch size {settings.batch_size})...")

        try:
            # Auto-reduce batch size on OOM (matches Heretic's doubling approach but in reverse)
            while settings.batch_size >= 1:
                try:
                    yield ("", f"Phase 5/8: Detecting response prefix (batch size {settings.batch_size})...")
                    responses = model.get_responses_batched(
                        good_prompts[:100] + bad_prompts[:100]
                    )
                    break
                except (torch.cuda.OutOfMemoryError, Exception) as e:
                    if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                        new_bs = max(1, settings.batch_size // 2)
                        if new_bs == settings.batch_size:
                            raise
                        settings.batch_size = new_bs
                        yield ("", f"Phase 5/8: OOM — reducing batch size to {new_bs}")
                    else:
                        raise

            # Detect common response prefix (matches Heretic's run() exactly)
            model.response_prefix = commonprefix(responses).rstrip(" ")

            # Suppress CoT output (matches Heretic's run() exactly)
            if model.response_prefix.startswith("<think>"):
                model.response_prefix = "<think></think>"
            elif model.response_prefix.startswith("<|channel|>analysis<|message|>"):
                model.response_prefix = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
            elif model.response_prefix.startswith("<thought>"):
                model.response_prefix = "<thought></thought>"
            elif model.response_prefix.startswith("[THINK]"):
                model.response_prefix = "[THINK][/THINK]"

            if model.response_prefix:
                yield ("", f"Phase 5/8: Response prefix: {model.response_prefix!r}")

            # Compute refusal directions (matches Heretic's run() exactly)
            yield ("", "Phase 5/8: Extracting residuals for good prompts...")
            good_residuals = model.get_residuals_batched(good_prompts)

            yield ("", "Phase 5/8: Extracting residuals for bad prompts...")
            bad_residuals = model.get_residuals_batched(bad_prompts)

            good_means = good_residuals.mean(dim=0)
            bad_means = bad_residuals.mean(dim=0)

            refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

            # Free residuals memory
            del good_residuals, bad_residuals
            empty_cache()

        except torch.cuda.OutOfMemoryError:
            yield ("", "Out of GPU memory during direction computation. Try enabling 4-bit quantization or using a smaller model.")
            return
        except Exception as e:
            yield ("", f"Failed during direction computation: {e}")
            return

        elapsed = format_duration(time.time() - start_time)
        yield ("", f"Phase 5/8: Refusal directions computed ({elapsed})")

        # --- Phase 6/8: Set up evaluator ---
        yield ("", "Phase 6/8: Setting up evaluator...\nLoading evaluation prompts and computing baselines — this takes 1-2 min...")

        from heretic.evaluator import Evaluator

        try:
            evaluator = Evaluator(settings, model)
            base_refusals = getattr(evaluator, "base_refusals", None)
            n_bad = len(getattr(evaluator, "bad_prompts", []))
        except Exception as e:
            yield ("", f"Failed to set up evaluator: {e}")
            return

        elapsed = format_duration(time.time() - start_time)
        base_info = f"Base refusals: {base_refusals}/{n_bad}" if base_refusals is not None else ""
        yield ("", f"Phase 6/8: Evaluator ready ({elapsed})\n{base_info}")

        # --- Phase 7/8: Optuna optimization loop ---
        yield ("", f"Phase 7/8: Starting optimization — {n_trials} trials...")

        import optuna
        from optuna.exceptions import ExperimentalWarning
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=ExperimentalWarning)

        study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=settings.n_startup_trials,
                n_ei_candidates=128,
                multivariate=True,
            ),
            directions=["minimize", "minimize"],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discover ALL abliterable components across every layer.
        # Hybrid architectures (e.g. Qwen3.5 with DeltaNet + Attention layers)
        # have different components per layer type.  heretic's
        # get_abliterable_components() only checks layer 0, which causes a
        # KeyError in abliterate() when it encounters layers with extra
        # components.  We scan all layers and provide parameters for every
        # component seen anywhere.
        all_components = set()
        for _li in range(last_layer_index + 1):
            all_components.update(model.get_layer_modules(_li).keys())
        all_components = sorted(all_components)

        # --- Compute layer targeting ranges ---
        if target_output_layers_only:
            dir_idx_lo = 0.6 * last_layer_index
            dir_idx_hi = 0.9 * last_layer_index
            pos_lo = 0.7 * last_layer_index
            pos_hi = 1.0 * last_layer_index
            dist_hi = max(1.1, 0.3 * last_layer_index)
        elif layer_range_min > 0 or layer_range_max > 0:
            lr_min = max(0, layer_range_min)
            lr_max = min(layer_range_max, last_layer_index) if layer_range_max > 0 else last_layer_index
            dir_idx_lo = lr_min + 0.1 * (lr_max - lr_min)
            dir_idx_hi = lr_min + 0.9 * (lr_max - lr_min)
            pos_lo = float(lr_min)
            pos_hi = float(lr_max)
            dist_hi = max(1.1, 0.6 * (lr_max - lr_min))
        else:
            # Default: full range (same as original)
            dir_idx_lo = 0.4 * last_layer_index
            dir_idx_hi = 0.9 * last_layer_index
            pos_lo = 0.6 * last_layer_index
            pos_hi = 1.0 * last_layer_index
            dist_hi = max(1.1, 0.6 * last_layer_index)

        # Safeguards — ensure lo < hi
        dir_idx_lo = max(0.1, dir_idx_lo)
        dir_idx_hi = max(dir_idx_lo + 0.1, dir_idx_hi)
        pos_lo = max(0.1, pos_lo)
        pos_hi = max(pos_lo + 0.1, pos_hi)
        dist_hi = max(1.1, dist_hi)

        best_refusals = float("inf")
        best_kl = float("inf")
        phase7_start = time.time()
        trial_index = 0
        kl_exceeded_count = 0
        consecutive_errors = 0
        last_error = None

        def objective(trial):
            """Optuna objective — matches Heretic's run() exactly."""
            nonlocal trial_index
            trial_index += 1

            direction_scope = trial.suggest_categorical(
                "direction_scope", ["global", "per layer"]
            )

            # Always suggest direction_index (multivariate TPE needs consistent params)
            direction_index = trial.suggest_float(
                "direction_index",
                dir_idx_lo,
                dir_idx_hi,
            )

            if direction_scope == "per layer":
                direction_index = None

            parameters = {}
            for component in all_components:
                max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
                max_weight_position = trial.suggest_float(
                    f"{component}.max_weight_position",
                    pos_lo,
                    pos_hi,
                )
                min_weight_frac = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
                min_weight_distance = trial.suggest_float(
                    f"{component}.min_weight_distance",
                    1.0,
                    dist_hi,
                )

                parameters[component] = AbliterationParameters(
                    max_weight=max_weight,
                    max_weight_position=max_weight_position,
                    min_weight=(min_weight_frac * max_weight),
                    min_weight_distance=min_weight_distance,
                )

            trial.set_user_attr("direction_index", direction_index)
            trial.set_user_attr(
                "parameters",
                {k: {"max_weight": v.max_weight, "max_weight_position": v.max_weight_position,
                      "min_weight": v.min_weight, "min_weight_distance": v.min_weight_distance}
                 for k, v in parameters.items()},
            )

            model.reset_model()
            model.abliterate(refusal_directions, direction_index, parameters)
            score, kl_divergence, refusals = evaluator.get_score()

            trial.set_user_attr("kl_divergence", kl_divergence)
            trial.set_user_attr("refusals", refusals)

            # KL ceiling enforcement — penalise trials that exceed the cap
            if kl_divergence > kl_ceiling:
                trial.set_user_attr("kl_exceeded", True)
                return (999.0, 999.0)

            return score

        n_bad_eval = len(getattr(evaluator, "bad_prompts", []))
        best_trial_num = 0
        improved_count = 0

        for trial_num in range(1, n_trials + 1):
            try:
                study.optimize(objective, n_trials=1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                yield (
                    _make_dashboard_html(
                        current_kl=None, best_kl=best_kl if best_kl != float("inf") else None,
                        best_refusals=best_refusals if best_refusals != float("inf") else None,
                        best_trial=best_trial_num, improvements=improved_count,
                        n_bad=n_bad_eval, kl_ceiling=kl_ceiling, refusal_target=refusal_target,
                        kl_exceeded_count=kl_exceeded_count,
                    ),
                    f"Phase 7/8: Trial {trial_num}/{n_trials} — OOM, skipping\n"
                    f"  Best so far: {best_refusals}/{n_bad_eval} refusals, KL: {best_kl:.4f}",
                )
                continue
            except Exception as e:
                consecutive_errors += 1
                last_error = str(e)
                yield ("", f"Phase 7/8: Trial {trial_num}/{n_trials} — Error: {e}")
                if consecutive_errors >= 3:
                    yield ("", (
                        f"Phase 7/8: Aborting — {consecutive_errors} consecutive trial failures.\n"
                        f"Last error: {last_error}\n\n"
                        "This model's architecture may not be compatible with abliteration."
                    ))
                    break
                continue

            consecutive_errors = 0  # reset on success

            # Extract latest trial results from user_attrs
            latest = study.trials[-1]
            kl = latest.user_attrs.get("kl_divergence", float("inf"))
            refs = latest.user_attrs.get("refusals", float("inf"))
            kl_over = latest.user_attrs.get("kl_exceeded", False)

            if kl_over:
                kl_exceeded_count += 1

            new_best = ""
            if not kl_over and _is_better(refs, kl, best_refusals, best_kl, refusal_target):
                best_refusals = refs
                best_kl = kl
                best_trial_num = trial_num
                improved_count += 1
                new_best = "  ** New best! **\n"

            # Progress bar
            pct = trial_num / n_trials
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

            # ETA calculation
            elapsed_phase7 = time.time() - phase7_start
            avg_per_trial = elapsed_phase7 / trial_num
            remaining = avg_per_trial * (n_trials - trial_num)
            total_elapsed = format_duration(time.time() - start_time)
            eta = format_duration(remaining)

            # KL status label
            if kl_over:
                kl_status = f"KL: {kl:.4f} [EXCEEDED CEILING]"
            elif kl < 0.8:
                kl_status = f"KL: {kl:.4f} [OK]"
            elif kl <= 1.5:
                kl_status = f"KL: {kl:.4f} [WARN]"
            else:
                kl_status = f"KL: {kl:.4f} [DANGER]"

            # Best KL status label
            if best_kl != float("inf"):
                if best_kl < 0.8:
                    best_kl_status = f"KL: {best_kl:.4f} [OK]"
                elif best_kl <= 1.5:
                    best_kl_status = f"KL: {best_kl:.4f} [WARN]"
                else:
                    best_kl_status = f"KL: {best_kl:.4f} [DANGER]"
            else:
                best_kl_status = "KL: ---"

            exceeded_note = f"  KL ceiling exceeded: {kl_exceeded_count} trials discarded\n" if kl_exceeded_count > 0 else ""

            dashboard = _make_dashboard_html(
                current_kl=kl, best_kl=best_kl if best_kl != float("inf") else None,
                best_refusals=best_refusals if best_refusals != float("inf") else None,
                best_trial=best_trial_num, improvements=improved_count,
                n_bad=n_bad_eval, kl_ceiling=kl_ceiling, refusal_target=refusal_target,
                kl_exceeded_count=kl_exceeded_count,
            )

            yield (
                dashboard,
                f"Phase 7/8: Optimization — Trial {trial_num}/{n_trials}\n"
                f"  {bar} {pct:.0%}\n"
                f"{new_best}"
                f"  This trial: {refs}/{n_bad_eval} refusals, {kl_status}\n"
                f"  Best so far: {best_refusals}/{n_bad_eval} refusals, {best_kl_status} (trial #{best_trial_num})\n"
                f"  Improvements: {improved_count} | Avg/trial: {format_duration(avg_per_trial)}\n"
                f"{exceeded_note}"
                f"  Elapsed: {total_elapsed} | ETA: ~{eta}",
            )

        elapsed = format_duration(time.time() - start_time)
        yield (
            _make_dashboard_html(
                current_kl=best_kl if best_kl != float("inf") else None,
                best_kl=best_kl if best_kl != float("inf") else None,
                best_refusals=best_refusals if best_refusals != float("inf") else None,
                best_trial=best_trial_num, improvements=improved_count,
                n_bad=n_bad_eval, kl_ceiling=kl_ceiling, refusal_target=refusal_target,
                kl_exceeded_count=kl_exceeded_count,
            ),
            f"Phase 7/8: Optimization complete ({elapsed})\n"
            f"  Best: {best_refusals}/{n_bad_eval} refusals, KL: {best_kl:.4f} (trial #{best_trial_num})\n"
            f"  Improvements found: {improved_count}/{n_trials} trials"
            + (f"\n  KL ceiling exceeded: {kl_exceeded_count} trials discarded" if kl_exceeded_count > 0 else ""),
        )

        # --- Phase 8/8: Apply best params + save ---
        yield ("", "Phase 8/8: Applying best parameters and saving model...")

        try:
            # Find best trial — exclude KL-exceeded trials, use refusal-target-aware sorting
            valid_trials = [
                t for t in study.trials
                if t.user_attrs.get("refusals") is not None
                and not t.user_attrs.get("kl_exceeded", False)
            ]

            if not valid_trials:
                # No trial met the KL ceiling — fall back to the one with lowest KL
                valid_trials = [
                    t for t in study.trials
                    if t.user_attrs.get("refusals") is not None
                ]
                yield ("", "WARNING: No trial met the KL ceiling. Using the trial with lowest KL divergence.")

            if not valid_trials:
                import optuna
                n_fail = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
                msg = f"No completed trials found. Cannot save model.\n  Total trials: {len(study.trials)}, Failed: {n_fail}"
                if last_error:
                    msg += f"\n  Last error: {last_error}"
                yield ("", msg)
                return

            best_trial = min(valid_trials, key=lambda t: _trial_sort_key(t, refusal_target))

            # Update tracking from the actual best trial
            best_kl = best_trial.user_attrs["kl_divergence"]
            best_refusals = best_trial.user_attrs["refusals"]

            # Restore best trial's parameters (matches Heretic's restoration code exactly)
            model.reset_model()
            model.abliterate(
                refusal_directions,
                best_trial.user_attrs["direction_index"],
                {
                    k: AbliterationParameters(**v)
                    for k, v in best_trial.user_attrs["parameters"].items()
                },
            )

            # Save merged model
            output_path.mkdir(parents=True, exist_ok=True)
            yield ("", "Phase 8/8: Merging LoRA weights and saving...")

            merged = model.get_merged_model()
            merged.save_pretrained(str(output_path))
            del merged
            empty_cache()
            model.tokenizer.save_pretrained(str(output_path))

            model_size = get_model_size(output_path)
            total_time = format_duration(time.time() - start_time)

            # KL quality assessment
            if best_kl < 0.01:
                kl_grade = "Excellent — minimal capability loss"
            elif best_kl < 0.1:
                kl_grade = "Good — slight capability shift"
            elif best_kl < 0.5:
                kl_grade = "Moderate — noticeable capability change"
            elif best_kl < 1.0:
                kl_grade = "High — significant capability impact"
            else:
                kl_grade = "Very high — model may be damaged"

            # Refusal assessment
            base_ref = base_refusals if base_refusals is not None else "?"
            if best_refusals == 0:
                ref_grade = "Perfect — zero refusals"
            elif best_refusals <= 3:
                ref_grade = "Excellent — near-zero refusals"
            elif best_refusals <= 10:
                ref_grade = "Good — minimal refusals remain"
            else:
                ref_grade = "Partial — some refusals remain"

            # Model health
            health_text, _ = _health_label(best_kl)
            if best_kl > 1.5:
                health_text += " — consider re-running with looser refusal target"

            # GPU info
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"

            # Layer targeting info
            layer_info = "output layers only" if target_output_layers_only else "all layers"
            if not target_output_layers_only and (layer_range_min > 0 or layer_range_max > 0):
                layer_info = f"layers {layer_range_min}-{layer_range_max}"

            # Final dashboard
            final_dashboard = _make_dashboard_html(
                current_kl=best_kl, best_kl=best_kl,
                best_refusals=best_refusals, best_trial=best_trial_num,
                improvements=improved_count, n_bad=n_bad_eval,
                kl_ceiling=kl_ceiling, refusal_target=refusal_target,
                kl_exceeded_count=kl_exceeded_count,
            )

            yield (
                final_dashboard,
                f"Decensoring complete!\n"
                f"{'=' * 45}\n"
                f"\n"
                f"MODEL\n"
                f"  Source:      {model_path}\n"
                f"  Output:      {output_path}\n"
                f"  Size:        {format_size(model_size)}\n"
                f"\n"
                f"SETTINGS\n"
                f"  Mode:        {mode}\n"
                f"  Trials:      {n_trials}\n"
                f"  Batch size:  {settings.batch_size}\n"
                f"  Quantize:    {quantization}\n"
                f"  KL ceiling:  {kl_ceiling}\n"
                f"  Refusal tgt: {refusal_target}\n"
                f"  Layers:      {layer_info}\n"
                f"  GPU:         {gpu_name}\n"
                f"\n"
                f"RESULTS\n"
                f"  Refusals:    {base_ref} -> {best_refusals}/{n_bad_eval}  ({ref_grade})\n"
                f"  KL diverge:  {best_kl:.4f}  ({kl_grade})\n"
                f"  Model health: {health_text}\n"
                f"  Best trial:  #{best_trial_num}\n"
                f"  Improvements: {improved_count}/{n_trials} trials\n"
                + (f"  KL exceeded: {kl_exceeded_count} trials discarded\n" if kl_exceeded_count > 0 else "")
                + f"\n"
                f"TIMING\n"
                f"  Total:       {total_time}\n"
                f"  Avg/trial:   {format_duration((time.time() - phase7_start) / max(n_trials, 1))}\n"
                f"{'=' * 45}\n"
                f"Ready to download from the Models tab!",
            )

        except Exception as e:
            yield ("", f"Failed to save model: {e}")
            return

    except torch.cuda.OutOfMemoryError:
        yield ("", "Out of GPU memory. Try enabling 4-bit quantization or using a smaller model.")
    except Exception as e:
        yield ("", f"Decensoring failed: {e}")
    finally:
        # Always free GPU memory
        if model is not None:
            try:
                del model
            except Exception:
                pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _check_cuda() -> tuple[bool, str]:
    """Check CUDA availability."""
    from src.utils import check_cuda_available
    return check_cuda_available()
