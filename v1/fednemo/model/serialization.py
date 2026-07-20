"""Serialize records into text prompts and parse model output back to a label.

Dataset-generic: the label space is registered at load time from the dataset's
own labels (set_label_space), so nothing here is tied to a specific dataset.
Works for text->label classification (e.g. symptom text -> disease) and for
records that also carry structured fields.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..data.record import ClinicalRecord

# --------------------------------------------------------------------------- #
# Label space registry (set once from the dataset's labels)
# --------------------------------------------------------------------------- #
_LABEL_SPACE: List[str] = []


def set_label_space(labels: List[str]) -> None:
    """Register the sorted unique label set for the loaded dataset."""
    global _LABEL_SPACE
    _LABEL_SPACE = sorted({str(x).strip() for x in labels})


def get_label_space() -> List[str]:
    return list(_LABEL_SPACE)


def label_space_for(source: Optional[str] = None) -> List[str]:
    """Return the registered label space (source arg kept for call compatibility)."""
    return list(_LABEL_SPACE)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
_INSTRUCTION = (
    "You are a medical decision-support assistant. Based only on the information "
    "below, state the single most likely diagnosis from the allowed list, then "
    "briefly explain your reasoning grounded in the specific details given."
)


def _format_features(features: Dict[str, object]) -> str:
    lines = []
    for key, value in features.items():
        if value is None or value == "":
            continue
        lines.append(f"  - {key.replace('_', ' ')}: {value}")
    return "\n".join(lines) if lines else ""


def build_prompt(record: ClinicalRecord, label_space: List[str]) -> str:
    """Build the instruction+record prompt (without the target answer)."""
    label_str = ", ".join(label_space)
    feats = _format_features(record.features)
    feats_block = f"Structured fields:\n{feats}\n" if feats else ""
    free_text_block = f"Reported symptoms / notes:\n  {record.free_text}\n" if record.free_text else ""
    return (
        f"{_INSTRUCTION}\n"
        f"Allowed diagnoses: {label_str}\n\n"
        f"[PATIENT_RECORD]\n"
        f"{feats_block}"
        f"{free_text_block}"
        f"[/PATIENT_RECORD]\n\n"
        f"Answer:\n"
    )


_INSTRUCTION_PARAPHRASE = (
    "Act as a clinical triage assistant. Using only the case details provided, "
    "give the single most probable diagnosis from the permitted list, and justify "
    "your answer with reference to the concrete details shown."
)


def build_prompt_paraphrased(record: ClinicalRecord, label_space: List[str]) -> str:
    """A clinically-equivalent paraphrase of build_prompt: different wording and
    reordered content. Used to test semantic robustness (item 7)."""
    label_str = ", ".join(label_space)
    items = [(k, v) for k, v in record.features.items() if v is not None and v != ""]
    items = list(reversed(items))
    lines = [f"  * {k.replace('_', ' ')}: {v}" for k, v in items]
    fields_block = ("Case details:\n" + "\n".join(lines) + "\n") if lines else ""
    free_text_block = f"Clinical narrative:\n  {record.free_text}\n" if record.free_text else ""
    return (
        f"{_INSTRUCTION_PARAPHRASE}\n"
        f"Permitted diagnoses: {label_str}\n\n"
        f"[PATIENT_RECORD]\n"
        f"{fields_block}"
        f"{free_text_block}"
        f"[/PATIENT_RECORD]\n\n"
        f"Answer:\n"
    )


def build_target(record: ClinicalRecord) -> str:
    """Supervised target completion for a training record."""
    return f"Diagnosis: {record.label}."


def parse_model_output(text: str, label_space: List[str]) -> Dict[str, object]:
    """Extract the predicted label from free-form output.

    Deterministic: after an optional 'Diagnosis:'/'Answer:' cue, find the allowed
    label whose text appears earliest (handles multi-word labels like
    'Bronchial Asthma' or 'urinary tract infection'). Case-insensitive.
    """
    lower = text.lower()
    m = re.search(r"(?:diagnosis|diagnostic class|answer)\s*[:\-]?\s*", lower)
    region = lower[m.end():] if m else lower

    best: Optional[str] = None
    best_pos = len(region) + 1
    for lab in label_space:
        pos = region.find(lab.lower())
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best = lab
    if best is None:  # fallback: search whole text
        for lab in label_space:
            if lab.lower() in lower:
                best = lab
                break
    return {"predicted_label": best, "raw_output": text.strip()}
