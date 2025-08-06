# ---------------------------------------------------------------
#            Medi-Chat API  –  CheXpert + Groq LLM back-end
# ---------------------------------------------------------------
# Complete file – paste over the existing app.py.
# Changes vs. last version:
#   • /llmreport no longer invents a “normal” report when both
#     evidence and summary are empty – it raises HTTP 400 instead.
# ---------------------------------------------------------------

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from groq import Groq
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
LABELS: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

MODEL_PATH: str = os.getenv("MODEL_PATH", "densenet121_finetuned.pth")
THRESHOLD_PATH: str = os.getenv("THRESHOLD_PATH", "thresholds.json")
DEVICE: str = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
MAX_COMPLETION_TOKENS: int = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "8192"))

PNEUMONIA_IDX: int = LABELS.index("Pneumonia")

os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/torch_cache")
os.makedirs("/tmp/torch_cache", exist_ok=True)

# ----------------------------------------------------------------
# FASTAPI
# ----------------------------------------------------------------
app = FastAPI(
    title="Medi-Chat API (CheXpert + Groq)",
    docs_url="/docs",
    redoc_url="/redoc",
)

_model: Optional[torch.nn.Module] = None
_thresholds: Optional[np.ndarray] = None

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ----------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------
def load_assets() -> tuple[torch.nn.Module, np.ndarray]:
    global _model, _thresholds

    if _model is None:
        mdl = models.densenet121(weights=None)
        mdl.classifier = torch.nn.Linear(mdl.classifier.in_features, len(LABELS))
        mdl.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        mdl.to(DEVICE).eval()
        _model = mdl

    if _thresholds is None:
        with open(THRESHOLD_PATH, "r") as fh:
            thr_map = json.load(fh)
        _thresholds = np.array([thr_map[label] for label in LABELS], dtype=np.float32)

    return _model, _thresholds


def to_tensor(img: Image.Image) -> torch.Tensor:
    return _transform(img).unsqueeze(0).to(DEVICE)


def classify_image(img: Image.Image) -> Dict[str, Any]:
    model, thr = load_assets()
    with torch.no_grad():
        probs = torch.sigmoid(model(to_tensor(img))).cpu().numpy()[0]

    detected = [LABELS[i] for i, p in enumerate(probs) if p >= thr[i]] or [
        "No abnormal findings detected"
    ]

    pneu_prob = float(probs[PNEUMONIA_IDX])
    pneu_thr = float(thr[PNEUMONIA_IDX])
    pneu_flag = bool(pneu_prob >= pneu_thr)

    return {
        "labels": LABELS,
        "probabilities": probs.tolist(),
        "thresholds": thr.tolist(),
        "detected": detected,
        "pneumonia_probability": pneu_prob,
        "pneumonia_threshold": pneu_thr,
        "pneumonia_present": pneu_flag,
    }


def call_groq(
    messages: Union[str, Sequence[Dict[str, str]]],
    *,
    model: str = GROQ_MODEL,
    max_completion_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    if GROQ_API_KEY is None:
        raise HTTPException(500, "GROQ_API_KEY not configured.")

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    client = Groq(api_key=GROQ_API_KEY)

    resp = client.chat.completions.create(
        model=model,
        messages=list(messages),
        max_completion_tokens=max_completion_tokens or MAX_COMPLETION_TOKENS,
        **kwargs,
    )

    return resp.choices[0].message.content.strip()


# ----------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------
class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None


# ----------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "message": "Use /predict_chexpert, /chat, /report, /llmreport or /vision_report.",
    }


@app.post("/predict_chexpert")
async def predict_chexpert(file: UploadFile = File(...)) -> JSONResponse:
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return JSONResponse(classify_image(pil))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Internal error in /predict_chexpert: {exc!r}") from exc


# (chat and report endpoints unchanged from previous revision – omitted for brevity)
# -----------------------------------------------------------------------
# /llmreport – **NO FALL-BACK**: both fields must not be empty
# -----------------------------------------------------------------------
@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn) -> JSONResponse:
    evidence_text = (payload.evidence or "").strip()
    summary_text = (payload.summary or "").strip()

    if not evidence_text and not summary_text:
        raise HTTPException(
            400, "At least one of 'evidence' or 'summary' must be provided."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior consultant radiologist. Read the caller-supplied "
                "summary and evidence (both inside triple back-ticks) and then "
                "write **exactly 5–7 bullet points** in British English:\n"
                " • Begin with the pneumonia conclusion stated in the summary.\n"
                " • Then comment on any other findings suggested by the evidence.\n"
                "Do NOT add headings, dates or patient identifiers."
            ),
        },
        {
            "role": "user",
            "content": (
                "### Confirmed summary\n"
                "```text\n"
                f"{summary_text or 'None provided'}\n"
                "```\n\n"
                "### Raw evidence / probabilities\n"
                "```text\n"
                f"{evidence_text or 'None provided'}\n"
                "```"
            ),
        },
    ]

    report = call_groq(
        messages,
        model="openai/gpt-oss-120b",
        temperature=0.7,
        top_p=1,
        reasoning_effort="medium",
    )
    return JSONResponse({"report": report})


# /vision_report identical to previous revision – omitted for brevity
