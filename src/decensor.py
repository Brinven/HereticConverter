"""Core decensoring pipeline wrapping Heretic's abliteration engine."""

import os
import sys
import time
import warnings
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
    if quantization == "8-bit (bitsandbytes)":
        quant = QuantizationMethod.INT8
    elif quantization == "4-bit (bitsandbytes)":
        quant = QuantizationMethod.INT4

    kwargs = {
        "model": model_path,
        "n_trials": n_trials,
        "max_response_length": max_response_length,
        # Always set a non-zero batch_size — Heretic's batchify() crashes with 0.
        # If user selected auto (0), start with 32; we'll auto-tune down on OOM later.
        "batch_size": batch_size if batch_size > 0 else 32,
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


def run_decensor(
    model_path: str,
    output_name: str,
    mode: str,
    n_trials: int,
    quantization: str,
    batch_size: int,
    max_response_length: int,
    output_dir: str = "./models",
) -> Generator[str, None, None]:
    """Run the decensoring pipeline, yielding progress strings.

    This is a generator function. Each yield pushes a status update to the
    Gradio output Textbox in real time.
    """
    start_time = time.time()
    model = None

    try:
        import torch
        import torch.nn.functional as F

        # --- Phase 1/8: Validate inputs ---
        yield "Phase 1/8: Validating inputs..."

        if not model_path or not model_path.strip():
            yield "❌ Model path cannot be empty."
            return

        model_path = model_path.strip()

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
            yield f"Output already exists, using: {output_name}"

        cuda_available, cuda_info = _check_cuda()
        if not cuda_available:
            yield f"❌ No CUDA GPU detected. Heretic requires an NVIDIA GPU with CUDA support.\nDetails: {cuda_info}"
            return

        # Match Heretic's run() setup
        torch.set_grad_enabled(False)
        torch._dynamo.config.cache_size_limit = 64

        import transformers
        transformers.logging.set_verbosity_error()

        yield f"Phase 1/8: Validated. GPU: {cuda_info}"

        # --- Phase 2/8: Construct Settings ---
        yield "Phase 2/8: Configuring settings..."

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
            yield f"❌ Failed to create settings: {e}"
            return

        yield f"Phase 2/8: Settings configured — {mode} mode, {n_trials} trials"

        # --- Phase 3/8: Load model ---
        yield f"Phase 3/8: Loading model '{model_path}'...\nThis may take 30-60 seconds and will download the model if not cached."

        from heretic.model import Model, AbliterationParameters

        try:
            model = Model(settings)
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                yield f"❌ Model '{model_path}' was not found on HuggingFace. Please check the model path."
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                yield "❌ Network error. Please check your internet connection and try again."
            else:
                yield f"❌ Failed to load model: {error_msg}"
            return

        elapsed = format_duration(time.time() - start_time)
        yield f"Phase 3/8: Model loaded ({elapsed})"

        # --- Phase 4/8: Load prompt datasets ---
        yield "Phase 4/8: Loading prompt datasets..."

        from heretic.utils import load_prompts, empty_cache

        try:
            good_prompts = load_prompts(settings, settings.good_prompts)
            bad_prompts = load_prompts(settings, settings.bad_prompts)
        except Exception as e:
            yield f"❌ Failed to load prompt datasets: {e}"
            return

        yield f"Phase 4/8: Loaded {len(good_prompts)} good + {len(bad_prompts)} bad prompts"

        # --- Phase 5/8: Detect prefix + compute refusal directions ---
        yield f"Phase 5/8: Computing refusal directions (batch size {settings.batch_size})..."

        try:
            # Auto-reduce batch size on OOM (matches Heretic's doubling approach but in reverse)
            while settings.batch_size >= 1:
                try:
                    yield f"Phase 5/8: Detecting response prefix (batch size {settings.batch_size})..."
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
                        yield f"Phase 5/8: OOM — reducing batch size to {new_bs}"
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
                yield f"Phase 5/8: Response prefix: {model.response_prefix!r}"

            # Compute refusal directions (matches Heretic's run() exactly)
            yield "Phase 5/8: Extracting residuals for good prompts..."
            good_residuals = model.get_residuals_batched(good_prompts)

            yield "Phase 5/8: Extracting residuals for bad prompts..."
            bad_residuals = model.get_residuals_batched(bad_prompts)

            good_means = good_residuals.mean(dim=0)
            bad_means = bad_residuals.mean(dim=0)

            refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

            # Free residuals memory
            del good_residuals, bad_residuals
            empty_cache()

        except torch.cuda.OutOfMemoryError:
            yield "❌ Out of GPU memory during direction computation. Try enabling 4-bit quantization or using a smaller model."
            return
        except Exception as e:
            yield f"❌ Failed during direction computation: {e}"
            return

        elapsed = format_duration(time.time() - start_time)
        yield f"Phase 5/8: Refusal directions computed ({elapsed})"

        # --- Phase 6/8: Set up evaluator ---
        yield "Phase 6/8: Setting up evaluator...\nLoading evaluation prompts and computing baselines — this takes 1-2 min..."

        from heretic.evaluator import Evaluator

        try:
            evaluator = Evaluator(settings, model)
            base_refusals = getattr(evaluator, "base_refusals", None)
            n_bad = len(getattr(evaluator, "bad_prompts", []))
        except Exception as e:
            yield f"❌ Failed to set up evaluator: {e}"
            return

        elapsed = format_duration(time.time() - start_time)
        base_info = f"Base refusals: {base_refusals}/{n_bad}" if base_refusals is not None else ""
        yield f"Phase 6/8: Evaluator ready ({elapsed})\n{base_info}"

        # --- Phase 7/8: Optuna optimization loop ---
        yield f"Phase 7/8: Starting optimization — {n_trials} trials..."

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
        best_refusals = float("inf")
        best_kl = float("inf")
        phase7_start = time.time()
        trial_index = 0

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
                0.4 * last_layer_index,
                0.9 * last_layer_index,
            )

            if direction_scope == "per layer":
                direction_index = None

            parameters = {}
            for component in model.get_abliterable_components():
                max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
                max_weight_position = trial.suggest_float(
                    f"{component}.max_weight_position",
                    0.6 * last_layer_index,
                    1.0 * last_layer_index,
                )
                min_weight_frac = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
                min_weight_distance = trial.suggest_float(
                    f"{component}.min_weight_distance",
                    1.0,
                    max(1.1, 0.6 * last_layer_index),
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
                    f"Phase 7/8: Trial {trial_num}/{n_trials} — OOM, skipping\n"
                    f"  Best so far: {best_refusals}/{n_bad_eval} refusals, KL: {best_kl:.4f}"
                )
                continue
            except Exception as e:
                yield f"Phase 7/8: Trial {trial_num}/{n_trials} — Error: {e}"
                continue

            # Extract latest trial results from user_attrs
            latest = study.trials[-1]
            kl = latest.user_attrs.get("kl_divergence", float("inf"))
            refs = latest.user_attrs.get("refusals", float("inf"))

            new_best = ""
            if refs < best_refusals or (refs == best_refusals and kl < best_kl):
                best_refusals = refs
                best_kl = kl
                best_trial_num = trial_num
                improved_count += 1
                new_best = "  ** New best! **\n"

            # Progress bar
            pct = trial_num / n_trials
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)

            # ETA calculation
            elapsed_phase7 = time.time() - phase7_start
            avg_per_trial = elapsed_phase7 / trial_num
            remaining = avg_per_trial * (n_trials - trial_num)
            total_elapsed = format_duration(time.time() - start_time)
            eta = format_duration(remaining)
            trial_time = format_duration(time.time() - phase7_start - (avg_per_trial * (trial_num - 1))) if trial_num > 1 else format_duration(elapsed_phase7)

            yield (
                f"Phase 7/8: Optimization — Trial {trial_num}/{n_trials}\n"
                f"  {bar} {pct:.0%}\n"
                f"{new_best}"
                f"  This trial: {refs}/{n_bad_eval} refusals, KL: {kl:.4f}\n"
                f"  Best so far: {best_refusals}/{n_bad_eval} refusals, KL: {best_kl:.4f} (trial #{best_trial_num})\n"
                f"  Improvements: {improved_count} | Avg/trial: {format_duration(avg_per_trial)}\n"
                f"  Elapsed: {total_elapsed} | ETA: ~{eta}"
            )

        elapsed = format_duration(time.time() - start_time)
        yield (
            f"Phase 7/8: Optimization complete ({elapsed})\n"
            f"  Best: {best_refusals}/{n_bad_eval} refusals, KL: {best_kl:.4f} (trial #{best_trial_num})\n"
            f"  Improvements found: {improved_count}/{n_trials} trials"
        )

        # --- Phase 8/8: Apply best params + save ---
        yield "Phase 8/8: Applying best parameters and saving model..."

        try:
            # Find best trial (minimize refusals first, then KL) — matches Heretic's Pareto logic
            completed_trials = [t for t in study.trials if t.user_attrs.get("refusals") is not None]
            best_trial = min(
                completed_trials,
                key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]),
            )

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
            yield "Phase 8/8: Merging LoRA weights and saving..."

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

            # GPU info
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"

            yield (
                f"✅ Decensoring complete!\n"
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
                f"  GPU:         {gpu_name}\n"
                f"\n"
                f"RESULTS\n"
                f"  Refusals:    {base_ref} -> {best_refusals}/{n_bad_eval}  ({ref_grade})\n"
                f"  KL diverge:  {best_kl:.4f}  ({kl_grade})\n"
                f"  Best trial:  #{best_trial_num}\n"
                f"  Improvements: {improved_count}/{n_trials} trials\n"
                f"\n"
                f"TIMING\n"
                f"  Total:       {total_time}\n"
                f"  Avg/trial:   {format_duration((time.time() - phase7_start) / max(n_trials, 1))}\n"
                f"{'=' * 45}\n"
                f"Ready to download from the Models tab!"
            )

        except Exception as e:
            yield f"❌ Failed to save model: {e}"
            return

    except torch.cuda.OutOfMemoryError:
        yield "❌ Out of GPU memory. Try enabling 4-bit quantization or using a smaller model."
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            yield f"❌ Model '{model_path}' was not found on HuggingFace. Please check the model path."
        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
            yield "❌ Network error. Please check your internet connection and try again."
        else:
            yield f"❌ Decensoring failed: {error_msg}"
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
