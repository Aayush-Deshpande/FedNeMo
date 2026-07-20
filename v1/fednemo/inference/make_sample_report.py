"""Render a synthetic (but realistic) cardiac report image to exercise the
nemotron-parse API end-to-end. NOTE: this is a SYNTHETIC rendered report, not a
genuine scanned document - used to validate the API response shape + field
mapping. Swap in a real scan for a production test.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPORT_LINES = [
    "CITY CARDIOLOGY CENTER",
    "Cardiac Risk Assessment Report",
    "-------------------------------------------",
    "Patient Name: [REDACTED]        MRN: 00-42-17",
    "Age: 61 years            Sex: male",
    "",
    "Chest pain type: atypical angina",
    "Resting blood pressure: 138 mmHg",
    "Serum cholesterol: 244 mg/dl",
    "Fasting blood sugar > 120: yes",
    "Resting ECG: left ventricular hypertrophy",
    "Max heart rate achieved: 142",
    "Exercise-induced angina: no",
    "ST depression (oldpeak): 1.8",
    "ST slope: flat",
    "Number of major vessels: 1",
    "Thalassemia: reversible defect",
    "",
    "Impression: Intermediate cardiac risk. Recommend",
    "stress testing and lipid management.",
]


def make_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 800, 620
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        bold = ImageFont.truetype("arialbd.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
        bold = font
    y = 20
    for i, line in enumerate(REPORT_LINES):
        f = bold if i < 2 else font
        draw.text((30, y), line, fill="black", font=f)
        y += 30
    img.save(path)
    return path


if __name__ == "__main__":
    from ..config import ARTIFACTS_DIR
    out = make_report(ARTIFACTS_DIR / "sample_report.png")
    print("wrote", out)
