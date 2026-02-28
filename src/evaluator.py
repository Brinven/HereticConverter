"""Model evaluation and chat testing wrapping Heretic's Evaluator."""

import sys
import time
from typing import Generator

from src.utils import format_duration


def evaluate_model(
    model_path: str,
    quantization: str = "None",
) -> Generator[str, None, None]:
    """Evaluate a model's refusal rate and KL divergence.

    Yields progress strings for real-time status updates in Gradio.
    """
    model = None
    try:
        if not model_path or not model_path.strip():
            yield "Please provide a model path."
            return

        model_path = model_path.strip()

        yield "Loading model..."

        from heretic.config import Settings, QuantizationMethod
        from heretic.model import Model

        quant = None
        if quantization == "4-bit (bitsandbytes)":
            quant = QuantizationMethod.BNB_4BIT

        kwargs = {"model": model_path, "trust_remote_code": True}
        if quant is not None:
            kwargs["quantization"] = quant

        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            settings = Settings(**kwargs)
        finally:
            sys.argv = original_argv

        start_time = time.time()

        try:
            model = Model(settings)
        except Exception as e:
            yield f"Failed to load model: {e}"
            return

        load_time = format_duration(time.time() - start_time)
        yield f"Model loaded ({load_time}). Setting up evaluator..."

        from heretic.evaluator import Evaluator

        try:
            evaluator = Evaluator(settings, model)
        except Exception as e:
            yield f"Failed to set up evaluator: {e}"
            return

        yield "Computing evaluation scores..."

        try:
            score = evaluator.get_score()
            # score is ((kld_score, refusals_score), kl_divergence, refusals)
            kl_divergence = score[1]
            refusals = score[2]
            total_time = format_duration(time.time() - start_time)

            yield (
                f"Evaluation complete ({total_time})\n\n"
                f"Model: {model_path}\n"
                f"Refusals: {refusals}\n"
                f"KL Divergence: {kl_divergence:.4f}\n"
                f"Quantization: {quantization}"
            )
        except Exception as e:
            yield f"Evaluation failed: {e}"
            return

    except Exception as e:
        yield f"Evaluation error: {e}"
    finally:
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


def chat_with_model(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    quantization: str = "None",
) -> str:
    """Send a single prompt to a model and return its response.

    Non-generator — chat responses are fast enough for blocking calls.
    """
    model = None
    try:
        if not model_path or not model_path.strip():
            return "Please provide a model path."
        if not prompt or not prompt.strip():
            return "Please enter a prompt."

        model_path = model_path.strip()

        from heretic.config import Settings, QuantizationMethod
        from heretic.model import Model

        quant = None
        if quantization == "4-bit (bitsandbytes)":
            quant = QuantizationMethod.BNB_4BIT

        kwargs = {
            "model": model_path,
            "max_response_length": max_tokens,
            "trust_remote_code": True,
        }
        if quant is not None:
            kwargs["quantization"] = quant

        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            settings = Settings(**kwargs)
        finally:
            sys.argv = original_argv

        model = Model(settings)

        responses = model.get_responses([prompt])
        if responses and responses[0]:
            return responses[0]
        else:
            return "Model returned an empty response."

    except Exception as e:
        return f"Chat failed: {e}"
    finally:
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
