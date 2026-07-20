"""Inference entrypoint.

Flow (per spec):
  scanned report image
    -> [single network call] nemotron-parse -> transcribed text/markdown
    -> [deterministic] field_mapping -> normalized schema dict
    -> [deterministic] serialization -> structured prompt
    -> [local] trained global Nemotron -> answer
    -> parse into {primary answer, other findings, grounded explanation}

The doctor's question (if any) is appended so the model addresses it while also
surfacing incidental clinically relevant findings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config import ARTIFACTS_DIR
from ..model.nemotron_local import LoadedModel, generate, load_nemotron
from ..model.serialization import build_prompt, label_space_for, parse_model_output
from .field_mapping import build_record_from_mapping, map_parsed_to_schema
from .nemotron_parse_client import parse_document_image

logger = logging.getLogger("fednemo.infer")


@dataclass
class InferenceResult:
    predicted_label: Optional[str]
    answer: str
    mapped_fields: Dict[str, object]
    parse_status: str
    raw_model_output: str
    parse_error: Optional[str] = None


def _load_trained_model(adapter_path: Optional[Path]) -> LoadedModel:
    lm = load_nemotron(attach_lora=True)
    path = adapter_path or (ARTIFACTS_DIR / "global_adapter.pt")
    if path.exists():
        from ..federated.fedrand import load_adapter_state
        state = torch.load(path, map_location="cpu")
        load_adapter_state(lm.model, state)
        logger.info("Loaded trained global adapter from %s", path)
    else:
        logger.warning("No trained adapter at %s; using base model + fresh LoRA.", path)
    return lm


def run_inference(
    image_path: str,
    question: str = "",
    source_schema: str = "uci",
    api_key: Optional[str] = None,
    adapter_path: Optional[Path] = None,
    lm: Optional[LoadedModel] = None,
) -> InferenceResult:
    # 1. single external call: parse the scanned image
    parsed = parse_document_image(image_path, api_key=api_key)
    if parsed.status != "ok":
        logger.error("nemotron-parse failed: %s", parsed.error)
        return InferenceResult(
            predicted_label=None, answer="", mapped_fields={},
            parse_status="error", raw_model_output="", parse_error=parsed.error,
        )

    # 2. deterministic field mapping -> training schema
    mapped = map_parsed_to_schema(parsed.raw_content)
    logger.info("Mapped %d fields from parsed report.", len(mapped))

    # 3. serialize into a prompt for the trained model
    record = build_record_from_mapping(mapped, source=source_schema, free_text="")
    label_space = label_space_for(source_schema)
    base_prompt = build_prompt(record, label_space)
    if question.strip():
        base_prompt += f"Doctor's question: {question.strip()}\n"
    base_prompt += (
        "Provide: (1) the primary answer, (2) other clinically relevant findings, "
        "(3) a plain-language explanation grounded in the values above.\n"
    )

    # 4. local inference with the trained global model
    if lm is None:
        lm = _load_trained_model(adapter_path)
    output = generate(lm, base_prompt, max_new_tokens=160)

    parsed_out = parse_model_output(output, label_space)
    return InferenceResult(
        predicted_label=parsed_out["predicted_label"],
        answer=output,
        mapped_fields=mapped,
        parse_status="ok",
        raw_model_output=output,
    )
