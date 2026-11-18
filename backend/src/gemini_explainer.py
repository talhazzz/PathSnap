"""Optional PathSnap AI Route Analyst powered by the official Google Gemini SDK (google-genai).

Strict safety and architecture rules:
- Optional: core PathSnap map matching works identically if GEMINI_API_KEY is unset or unavailable.
- Reads GEMINI_API_KEY only from the environment (with safe fallback lookup in backend/.env or .env).
- NEVER hardcodes, logs, prints, or returns the API key.
- NEVER sends full GPS coordinate arrays or ground-truth reference routes to Gemini.
- Handles missing SDK, missing key, invalid key, rate limits/quota, model errors, timeouts,
  network errors, and malformed responses gracefully without crashing FastAPI.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("pathsnap.gemini")

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

# Current free-tier compatible Gemini model (per official google-genai SDK guidelines)
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS = 25.0


def _read_key_from_dotenv_files() -> dict[str, str]:
    """Read key=value pairs from backend/.env and repo .env without external dependencies."""
    values: dict[str, str] = {}
    for env_path in (REPO_ROOT / ".env", BACKEND_DIR / ".env"):
        if not env_path.is_file():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    values[k] = v
        except Exception:
            continue
    return values


def _get_gemini_config() -> tuple[str | None, str]:
    """Resolve (api_key, model_name) securely without ever logging the key."""
    dotenv_vals = _read_key_from_dotenv_files()
    api_key = os.environ.get("GEMINI_API_KEY") or dotenv_vals.get("GEMINI_API_KEY")
    if api_key:
        api_key = api_key.strip()
        # Treat placeholder values as unconfigured
        if not api_key or api_key.upper().startswith("YOUR_") or api_key == "CHANGE_ME":
            api_key = None

    model_name = (
        os.environ.get("GEMINI_MODEL")
        or dotenv_vals.get("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()
    return api_key, model_name


def is_gemini_configured() -> dict[str, Any]:
    """Return non-sensitive Gemini readiness status (NEVER exposes key material)."""
    api_key, model_name = _get_gemini_config()
    try:
        from google import genai as _genai  # noqa: F401
        sdk_installed = True
    except Exception:
        sdk_installed = False

    return {
        "sdk_installed": sdk_installed,
        "api_key_configured": bool(api_key),
        "model": model_name,
        "ready": bool(sdk_installed and api_key),
    }


def _build_safe_prompt(summary: dict[str, Any]) -> str:
    """Build a compact, non-leaky prompt using ONLY summary statistics (never raw GPS or ground truth)."""
    lines = [
        f"Trip ID: {summary.get('trip_id', 'unknown')}",
        f"Selected Model: {summary.get('model_name', 'unknown')} ({summary.get('model_type', 'classifier')})",
        f"HMM/Viterbi Decoding Enabled: {summary.get('use_hmm', True)}",
        f"GPS Observations Processed: {summary.get('gps_points_count', 'N/A')}",
        f"Candidate Road Segments Scored: {summary.get('candidate_rows_count', 'N/A')}",
        f"Reconstructed Road Arcs: {summary.get('predicted_arc_count', 'N/A')}",
        f"Approximate Trajectory Distance (km): {summary.get('approx_distance_km', 'N/A')}",
        f"Trip Duration (s): {summary.get('duration_seconds', 'N/A')}",
        f"Predicted Route Directed Continuity: {summary.get('route_continuity', 'N/A')}",
        f"Classifier-Only Directed Continuity: {summary.get('classifier_only_route_continuity', 'N/A')}",
    ]
    timings = summary.get("timings_seconds") or {}
    if timings:
        lines.append(
            f"Timings (s): feature_extraction={timings.get('feature_extraction')}, "
            f"ml_inference={timings.get('classifier_inference')}, "
            f"hmm_viterbi={timings.get('hmm_viterbi_decoding')}, total={timings.get('total')}"
        )

    return (
        "You are PathSnap's AI Route Analyst. Explain this real map-matching result concisely "
        "(4-6 bullet points, under 180 words) for a technical evaluator.\n"
        "STRICT RULES:\n"
        "- Use ONLY the statistics provided below. Do NOT invent road names, city names, accuracy percentages, or metrics not listed.\n"
        "- Explain how ML candidate scoring and HMM/Viterbi sequence decoding work together to turn noisy GPS points into a continuous road sequence.\n"
        "- Highlight what the route continuity numbers indicate (especially comparing classifier-only vs HMM/Viterbi continuity).\n"
        "- Mention realistic limitations (e.g., urban GPS multipath noise, candidate radius 60 m, or GPS sampling gaps).\n\n"
        "ACTUAL TRIP SUMMARY:\n" + "\n".join(lines)
    )


def explain_trip_summary(
    summary: dict[str, Any],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Generate an optional natural-language explanation using Gemini Interactions API."""
    api_key, model_name = _get_gemini_config()

    if not api_key:
        return {
            "status": "unavailable",
            "available": False,
            "model": model_name,
            "explanation": None,
            "message": (
                "AI explanation is unavailable because GEMINI_API_KEY is not configured. "
                "Core map matching is still operational."
            ),
        }

    try:
        from google import genai
    except Exception:
        return {
            "status": "unavailable",
            "available": False,
            "model": model_name,
            "explanation": None,
            "message": (
                "AI explanation is unavailable because the google-genai SDK is not installed. "
                "Core map matching is still operational."
            ),
        }

    prompt = _build_safe_prompt(summary)
    system_instruction = (
        "You are a concise, factual machine learning and geospatial map-matching analyst. "
        "Never fabricate metrics, coordinates, road names, or ground-truth details."
    )

    def _invoke() -> str | None:
        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model=model_name,
            input=prompt,
            system_instruction=system_instruction,
            store=False,
        )
        text = getattr(interaction, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_invoke)
            explanation = future.result(timeout=timeout_seconds)

        if not explanation:
            return {
                "status": "error",
                "available": True,
                "model": model_name,
                "explanation": None,
                "message": "Gemini returned an empty or malformed response. Core map matching is unaffected.",
            }

        return {
            "status": "ok",
            "available": True,
            "model": model_name,
            "explanation": explanation,
            "message": "AI route analysis generated successfully.",
        }

    except concurrent.futures.TimeoutError:
        logger.warning("Gemini explanation timed out after %.1fs", timeout_seconds)
        return {
            "status": "timeout",
            "available": True,
            "model": model_name,
            "explanation": None,
            "message": f"Gemini request timed out after {timeout_seconds:.0f}s. Core map matching is unaffected.",
        }
    except Exception as exc:
        err_text = str(exc)
        low = err_text.lower()
        if "429" in low or "quota" in low or "rate" in low or "resource_exhausted" in low:
            user_msg = "Gemini API quota or rate limit reached. Core map matching is unaffected."
            status = "rate_limited"
        elif "401" in low or "403" in low or "api key" in low or "unauthenticated" in low:
            user_msg = "Gemini API key was rejected or lacks permission. Check GEMINI_API_KEY."
            status = "auth_error"
        elif "404" in low or "not found" in low or "model" in low:
            user_msg = f"Configured Gemini model '{model_name}' is unavailable for this account."
            status = "model_unavailable"
        else:
            user_msg = "Gemini service is temporarily unreachable. Core map matching is unaffected."
            status = "error"
        # Log error class only (never log exception payload if it might echo auth headers)
        logger.warning("Gemini explanation failed (%s): %s", status, type(exc).__name__)
        return {
            "status": status,
            "available": True,
            "model": model_name,
            "explanation": None,
            "message": user_msg,
        }
