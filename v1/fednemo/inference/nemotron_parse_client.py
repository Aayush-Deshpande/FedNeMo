"""Client for NVIDIA's hosted nemotron-parse model.

This is the ONLY outbound network call in the entire FedNeMo system. It is used
only at inference time, only to transcribe a scanned medical report IMAGE into
text/markdown. It is never used during training (training is CSV-only).

The endpoint is OpenAI-compatible (chat/completions). Per NVIDIA's docs the
request must use control tokens (NOT natural language) and greedy decoding with
a repetition penalty. The model returns text/markdown with embedded bounding-box
and semantic-class tokens; downstream field_mapping.py extracts clinical values
from that text deterministically.

IMPORTANT (honesty note): the exact shape of the returned `content` string cannot
be verified here without a live API call (the key was rotated by the user and not
re-supplied). This client is written against NVIDIA's published request/response
format. `field_mapping.py` therefore parses defensively.
"""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import requests

try:
    import certifi
    _DEFAULT_CA = certifi.where()
except Exception:  # certifi should be present (requests dep), but be safe
    _DEFAULT_CA = True

# Use the OS trust store when available. On this machine an AV/proxy (Norton)
# MITM-intercepts TLS with a root CA present in the Windows store but NOT in
# certifi's bundle, which otherwise causes CERTIFICATE_VERIFY_FAILED. truststore
# makes Python trust the OS store, resolving it cleanly (no insecure fallback).
try:
    import truststore
    truststore.inject_into_ssl()
    _TRUSTSTORE = True
except Exception:
    _TRUSTSTORE = False

from ..config import NEMOTRON_PARSE_MODEL, NEMOTRON_PARSE_URL, NVIDIA_API_KEY_ENV

logger = logging.getLogger("fednemo.parse")


def _tls_verify() -> Union[str, bool]:
    """Resolve the TLS verification setting.

    Defaults to certifi's CA bundle (fixes 'unable to get local issuer
    certificate' on machines whose system trust store isn't picked up).
    Env overrides:
      - FEDNEMO_CA_BUNDLE=<path>  : use a specific CA bundle (e.g. corporate proxy)
      - FEDNEMO_INSECURE_TLS=1    : disable verification (LAST RESORT, insecure)
    """
    if os.environ.get("FEDNEMO_INSECURE_TLS") == "1":
        logger.warning("TLS verification DISABLED via FEDNEMO_INSECURE_TLS=1 (insecure).")
        return False
    bundle = os.environ.get("FEDNEMO_CA_BUNDLE")
    if bundle:
        return bundle
    if _TRUSTSTORE:
        # OS trust store is injected globally; use it (don't force certifi bundle).
        return True
    return _DEFAULT_CA

# Recommended default prompt from NVIDIA docs: bbox + classes + markdown text.
CONTROL_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


@dataclass
class ParseResult:
    raw_content: str          # model's raw text/markdown output
    model: str
    status: str               # "ok" or "error"
    error: Optional[str] = None


def _encode_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime = _MIME.get(p.suffix.lower(), "image/png")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_document_image(
    image_path: str,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> ParseResult:
    """Send one image to the hosted nemotron-parse model and return its output.

    api_key: if None, read from the NVIDIA_API_KEY environment variable. The key
    is never logged or written to disk.
    """
    key = api_key or os.environ.get(NVIDIA_API_KEY_ENV)
    if not key:
        return ParseResult(
            raw_content="", model=NEMOTRON_PARSE_MODEL, status="error",
            error=f"No API key. Set the {NVIDIA_API_KEY_ENV} environment variable.",
        )

    try:
        data_url = _encode_image(image_path)
    except FileNotFoundError as exc:
        return ParseResult("", NEMOTRON_PARSE_MODEL, "error", str(exc))

    # VERIFIED against the hosted endpoint (2026-07): content must be IMAGE-ONLY.
    # Adding any text part returns HTTP 400 "The model does not support text
    # input." The model applies its parse tools automatically and returns the
    # result via a tool_call (function `markdown_bbox`), not message.content.
    payload = {
        "model": NEMOTRON_PARSE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    logger.info("Calling hosted nemotron-parse (the only network call in FedNeMo)...")
    try:
        resp = requests.post(
            NEMOTRON_PARSE_URL, headers=headers, json=payload,
            timeout=timeout, verify=_tls_verify(),
        )
    except requests.RequestException as exc:
        return ParseResult("", NEMOTRON_PARSE_MODEL, "error", f"request failed: {exc}")

    if resp.status_code != 200:
        return ParseResult(
            "", NEMOTRON_PARSE_MODEL, "error",
            f"HTTP {resp.status_code}: {resp.text[:400]}",
        )

    try:
        body = resp.json()
        choice = body["choices"][0]
        message = choice.get("message", {})
        content = message.get("content")
        if not content and message.get("tool_calls"):
            # VERIFIED shape: tool_calls[0].function.arguments is a JSON string of
            # [{"bbox":{...}, "text": "...", "type": "..."}, ...] blocks (possibly
            # double-nested as [[...]]). Concatenate the block texts into markdown.
            args = message["tool_calls"][0].get("function", {}).get("arguments", "")
            content = _blocks_to_text(args)
        content = content or ""
    except (KeyError, IndexError, ValueError) as exc:
        return ParseResult(
            "", NEMOTRON_PARSE_MODEL, "error",
            f"unexpected response shape: {exc}; body[:300]={resp.text[:300]}",
        )

    return ParseResult(raw_content=content, model=NEMOTRON_PARSE_MODEL, status="ok")


def _blocks_to_text(arguments: str) -> str:
    """Extract and join the `text` fields from the parse tool's JSON arguments.

    Robust to the observed double-nested list form ([[{...}]]) and to raw
    fallback (returns the original string if it isn't the expected JSON)."""
    import json as _json

    def _walk(node, out):
        if isinstance(node, dict):
            if "text" in node and isinstance(node["text"], str):
                out.append(node["text"])
            else:
                for v in node.values():
                    _walk(v, out)
        elif isinstance(node, list):
            for v in node:
                _walk(v, out)

    try:
        parsed = _json.loads(arguments)
    except (ValueError, TypeError):
        return arguments  # not JSON; hand raw string to field mapping
    texts: list = []
    _walk(parsed, texts)
    # normalize the model's "<br>" soft breaks to newlines for regex matching
    joined = "\n".join(texts)
    return joined.replace("<br>", "\n")
