"""Deterministic field mapping: nemotron-parse output text -> training schema.

This is plain, rule-based code (NO model call). It scans the transcribed report
text for clinical fields and normalizes them into the exact feature dictionary
shape the model was trained on (see model/serialization.py and the UCI/PTB-XL
loaders). Values not found are left as None; the model is asked to reason over
whatever is present.

Because nemotron-parse returns free-form markdown (not labeled key/value JSON),
mapping relies on keyword + regex extraction. Each rule is explicit and auditable.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Each entry: (schema_field, [regex patterns], caster)
# Patterns capture group 1 = the value. Case-insensitive, run over the raw text.


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(s: str) -> Optional[int]:
    f = _to_float(s)
    return None if f is None else int(round(f))


_NUM = r"([-+]?\d+(?:\.\d+)?)"

_RULES: List[Tuple[str, List[str], str]] = [
    ("age", [r"age[:\s]+" + _NUM, _NUM + r"\s*(?:years|yrs|y/o|yo)\b"], "int"),
    ("resting_bp_mmHg", [r"(?:resting\s+)?(?:blood pressure|bp|trestbps)[:\s]+" + _NUM], "float"),
    ("cholesterol_mg_dl", [r"(?:serum\s+)?(?:cholesterol|chol)[:\s]+" + _NUM], "float"),
    ("max_heart_rate", [r"(?:max(?:imum)?\s+heart\s+rate(?:\s+achieved)?|thalach|thalch|max hr)[:\s]+" + _NUM], "float"),
    ("st_depression_oldpeak", [r"(?:st depression(?:\s*\(oldpeak\))?|oldpeak)[:\s]+" + _NUM], "float"),
    ("num_major_vessels", [r"(?:number of major vessels|major vessels|ca)[:\s]+" + _NUM], "int"),
    # --- item 5: added coverage ---
    ("height_cm", [r"height[:\s]+" + _NUM], "float"),
    ("weight_kg", [r"weight[:\s]+" + _NUM], "float"),
]

_SEX_PATTERNS = [
    (r"\b(male|man|m)\b", "male"),
    (r"\b(female|woman|f)\b", "female"),
]

# Order matters: check "atypical" before "typical" ("typical angina" is a
# substring of "atypical angina").
_CHEST_PAIN = [
    (r"atypical angina", "atypical angina"),
    (r"non-?anginal", "non-anginal"),
    (r"asymptomatic", "asymptomatic"),
    (r"typical angina", "typical angina"),
]

# item 5: categorical fields matched to their allowed value spellings
_RESTING_ECG = [
    (r"left ventricular hypertrophy|lv hypertrophy|lvh", "lv hypertrophy"),
    (r"st-?t (?:wave )?abnormalit|st-t abnormality", "st-t abnormality"),
    (r"normal", "normal"),
]
_ST_SLOPE = [
    (r"upsloping", "upsloping"),
    (r"downsloping", "downsloping"),
    (r"flat", "flat"),
]
_THAL = [
    (r"reversible defect", "reversable defect"),  # UCI spelling variant
    (r"fixed defect", "fixed defect"),
    (r"normal", "normal"),
]

_BOOL_FIELDS = {
    "exercise_induced_angina": [r"exercise[- ]induced angina[:\s]+(yes|true|positive|no|false|negative)"],
    # require the value token so "> 120: yes" isn't matched as empty/false
    "fasting_blood_sugar_gt_120": [r"fasting blood sugar[^\n]*?120[^\n]*?(yes|true|elevated|high|positive|no|false|normal)"],
}

_POSITIVE = {"yes", "true", "positive", "elevated", "high"}


def _match_categorical(lower: str, cue: str, options: List[Tuple[str, str]]) -> Optional[str]:
    """Match a categorical field. If a `cue` (e.g. 'resting ecg') is present, search
    the remainder of that line first; otherwise search the whole text."""
    # try to scope to the cue's line for precision
    line_match = re.search(cue + r"[^\n]*", lower)
    scopes = []
    if line_match:
        scopes.append(line_match.group(0))
    scopes.append(lower)
    for scope in scopes:
        for pat, norm in options:
            if re.search(pat, scope):
                return norm
    return None


def map_parsed_to_schema(parsed_text: str) -> Dict[str, object]:
    """Extract a normalized clinical feature dict from parsed report text."""
    text = parsed_text or ""
    lower = text.lower()
    out: Dict[str, object] = {}

    for field, patterns, caster in _RULES:
        value = None
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                raw = m.group(1)
                value = _to_int(raw) if caster == "int" else _to_float(raw)
                break
        if value is not None:
            out[field] = value

    # sex
    for pat, norm in _SEX_PATTERNS:
        if re.search(r"sex[:\s]+" + pat, lower) or re.search(r"gender[:\s]+" + pat, lower):
            out["sex"] = norm
            break

    # chest pain type
    for pat, norm in _CHEST_PAIN:
        if re.search(pat, lower):
            out["chest_pain_type"] = norm
            break

    # item 5: categorical fields (resting ECG, ST slope, thalassemia)
    v = _match_categorical(lower, r"resting ecg", _RESTING_ECG)
    if v is not None:
        out["resting_ecg"] = v
    v = _match_categorical(lower, r"st slope", _ST_SLOPE)
    if v is not None:
        out["st_slope"] = v
    v = _match_categorical(lower, r"thalassemia|thal\b", _THAL)
    if v is not None:
        out["thalassemia"] = v

    # boolean-ish fields
    for field, patterns in _BOOL_FIELDS.items():
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                token = next((g for g in reversed(m.groups()) if g), "")
                out[field] = token.strip().lower() in _POSITIVE
                break

    return out


def build_record_from_mapping(
    mapped: Dict[str, object],
    source: str = "uci",
    record_id: str = "inference_input",
    free_text: str = "",
):
    """Wrap a mapped feature dict into a ClinicalRecord for prompt serialization.

    `label` is empty (unknown - that's what we're predicting).
    """
    from ..data.record import ClinicalRecord
    return ClinicalRecord(
        record_id=record_id,
        features=mapped,
        label="",
        free_text=free_text,
        source=source,
    )
