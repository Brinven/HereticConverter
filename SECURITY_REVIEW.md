# Security Review: Heretic Converter

**Date:** 2026-02-22
**Reviewer:** Claude Opus 4.6 (automated)
**Scope:** All application source files (app.py, src/decensor.py, src/evaluator.py, src/utils.py, run.bat)

## Result: No actionable vulnerabilities found.

Four potential findings were identified and investigated in depth; all were determined to be false positives or below the confidence threshold:

| # | Finding | Confidence | Reason for Exclusion |
|---|---------|-----------|----------------------|
| 1 | Zip Slip in `import_model_zip()` | 3/10 | Python's `zipfile.extractall()` has built-in path traversal protection since well before 3.10; all `..` components are stripped automatically |
| 2 | Path traversal via `output_name` | 3/10 | Technically real but unexploitable — attacker cannot control written file contents or filenames (only model weights), must wait 30-60 min for pipeline, and cannot overwrite existing files |
| 3 | HF token leak in error messages | 3/10 | `huggingface_hub.login()` never includes the token in exception string representations; common errors use hardcoded messages like "Invalid user token." |
| 4 | Pickle deserialization via local model path | 3/10 | Requires pre-existing filesystem write access (circular), modern transformers defaults to safetensors not pickle, and loading local paths is intentional functionality |

## Note

The UI label "Saved in this session only — not stored to disk" on the HF token field is misleading — `huggingface_hub.login()` does persist the token to `~/.cache/huggingface/token`. This is a UX accuracy issue, not a security vulnerability.
