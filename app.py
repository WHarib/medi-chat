
# ================================================================
#                    Medi-Chat API (CheXpert + Groq)
# ================================================================

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

# ------------------------------------------------
# SIZE / ENCODING LIMITS (Groq)
# ------------------------------------------------
MAX_B64_BYTES = 3_600_000   # safety margin under Groq's 4 MB base64 cap
MAX_PIXELS    = 33_177_600  # 33 megapixels

def make_data_url_under_limit(img_bytes: bytes, filename: str | None = None) -> str:
    """
    Convert arbitrary input image bytes into a JPEG data URL whose base64 payload
    is best-effort <= MAX_B64_BYTES and <= MAX_PIXELS.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Enforce resolution cap
    if (img.width * img.height) > MAX_PIXELS:
        scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))))

    # Try qualities at current size
    for quality in (90, 80, 70, 60, 50, 40):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue())
        if len(b64) <= MAX_B64_BYTES:
            return "data:image/jpeg;base64," + b64.decode()

    # If still too big, downscale once and retry
    img = img.resize((max(1, int(img.width * 0.8)),
                      max(1, int(img.height * 0.8))))
    for quality in (70, 60, 50, 40, 35):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue())
        if len(b64) <= MAX_B64_BYTES:
            return "data:image/jpeg;base64," + b64.decode()

    # Last resort
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
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

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

MAX_COMPLETION_TOKENS: int = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "8192"))

# Vision model for /analyse (Maverick)
GROQ_VISION_MODEL_MAVERICK: str = os.getenv(
    "GROQ_VISION_MODEL_MAVERICK",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

PNEUMONIA_IDX: int = LABELS.index("Pneumonia")

os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/torch_cache")
os.makedirs("/tmp/torch_cache", exist_ok=True)

# ------------------------------------------------
# FASTAPI
# ------------------------------------------------
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

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
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
        _thresholds = np.array([thr_map[lbl] for lbl in LABELS], dtype=np.float32)

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
    messages: Union[str, Sequence[Dict[str, Any]]],
    *,
    model: str = GROQ_MODEL,
    max_completion_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """Universal wrapper round Groq chat-completion."""
    if not GROQ_API_KEY:
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


# ------------------------------------------------
# Pydantic model
# ------------------------------------------------
class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None


# ------------------------------------------------
# ENDPOINTS
# ------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "message": "Use /predict_chexpert, /chat, /report, /llmreport, /vision_report, or /analyse.",
    }


# ---------- /predict_chexpert ------------------------------------------------
@app.post("/predict_chexpert")
async def predict_chexpert(file: UploadFile = File(...)) -> JSONResponse:
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return JSONResponse(classify_image(pil))
    except Exception as exc:
        raise HTTPException(500, f"Internal error in /predict_chexpert: {exc!r}") from exc


# ---------- /chat ------------------------------------------------------------
LABEL_SET = {"pneumonia", "no_evidence", "unsure"}
SYS_TEMPLATES = {
    "pneumonia": (
        "You are a senior consultant radiologist. The image shows pneumonia. "
        "Provide a confident description of its anatomical location, radiographic "
        "features and likely severity."
    ),
    "no_evidence": (
        "You are a senior consultant radiologist. The image shows **no evidence "
        "of pneumonia**. Provide a concise, reassuring statement in British "
        "English, highlighting clear lungs and normal mediastinal contours. "
        "Do **not** express diagnostic uncertainty or recommend further imaging."
    ),
    "unsure": (
        "You are a senior consultant radiologist. Findings are equivocal for "
        "pneumonia. Briefly state the uncertainty and outline sensible next steps."
    ),
}
USER_TEMPLATES = {
    "pneumonia": (
        "Assessment summary: This image demonstrates pneumonia.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Describe location and features."
    ),
    "no_evidence": (
        "Assessment summary: No evidence of pneumonia is seen on this image.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Reassure; do not express uncertainty."
    ),
    "unsure": (
        "Assessment summary: Findings are uncertain for pneumonia.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Suggest next steps."
    ),
}


def detect_label(text: str) -> str:
    t = (text or "").lower()
    if "pneum" in t:
        return "pneumonia"
    if "no" in t and ("evid" in t or "pneumonia" in t):
        return "no_evidence"
    if any(k in t for k in ("unsure", "uncertain", "equivoc")):
        return "unsure"
    return ""


def labels_from_any_json(blob: str) -> List[str]:
    try:
        data = json.loads(blob)
    except Exception:
        return []
    found: List[str] = []
    if isinstance(data, dict) and "final_label" in data:
        found.append(str(data["final_label"]))
    elif isinstance(data, list):
        for itm in data:
            if isinstance(itm, dict) and "final_label" in itm:
                found.append(str(itm["final_label"]))
    return found


@app.post("/chat")
async def chat_endpoint(
    request: Request,
    file: UploadFile = File(...),
    final_label: str = Form("", description="Optional explicit label"),
    other_models: str = Form(""),
) -> JSONResponse:
    # Quick image sanity-check
    try:
        Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}") from exc

    # Collect label hints
    form = await request.form()
    raw_candidates: List[str] = []

    if final_label:
        raw_candidates.append(final_label)

    json_blob = form.get("json", "")
    try:
        blob_obj = json.loads(json_blob)
        if isinstance(blob_obj, dict) and "final_label" in blob_obj:
            raw_candidates.append(str(blob_obj["final_label"]))
        elif isinstance(blob_obj, list):
            for itm in blob_obj:
                if isinstance(itm, dict) and "final_label" in itm:
                    raw_candidates.append(str(itm["final_label"]))
    except Exception:
        pass

    raw_candidates += [other_models, form.get("other_models", "")]
    raw_candidates += labels_from_any_json(other_models or "")
    raw_candidates += labels_from_any_json(form.get("other_models", "") or "")

    chosen_label = next(
        (detect_label(c) for c in raw_candidates if detect_label(c)), "unsure"
    )
    if chosen_label == "unsure" and all(not c.strip() for c in raw_candidates):
        chosen_label = "no_evidence"

    # Optional extra context
    try:
        ctx = json.loads(other_models or json_blob or "{}")
        if not isinstance(ctx, dict):
            ctx = {}
    except Exception:
        ctx = {}

    messages = [
        {"role": "system", "content": SYS_TEMPLATES[chosen_label]},
        {
            "role": "user",
            "content": USER_TEMPLATES[chosen_label].format(
                data=json.dumps(ctx, indent=2)
            ),
        },
    ]
    answer = call_groq(messages)
    return JSONResponse({"answer": answer})


# ---------- /report ----------------------------------------------------------
@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    final_label: str = Form(...),
) -> JSONResponse:
    lbl = detect_label(final_label)
    if lbl == "":
        raise HTTPException(
            400, "final_label must be pneumonia, no_evidence or unsure."
        )

    hdr_map = {
        "pneumonia": "This image demonstrates pneumonia.",
        "no_evidence": "No evidence of pneumonia is seen on this image.",
        "unsure": "Findings are uncertain for pneumonia.",
    }
    detail_map = {
        "pneumonia": "Draft 5–7 bullet points on pneumonia presence & features.",
        "no_evidence": "Draft 5–7 reassuring bullet points on normal CXR; no uncertainty.",
        "unsure": "Draft 5–7 bullet points on uncertainty & next steps.",
    }

    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Assessment Summary: {hdr_map[lbl]}\n\n"
        f"{detail_map[lbl]}"
    )
    report = call_groq(prompt)
    return JSONResponse({"report": report, "final_label": lbl})


# ---------- /llmreport -------------------------------------------------------
@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn) -> JSONResponse:
    evidence_text = (payload.evidence or "").strip()
    summary_text  = (payload.summary  or "").strip()

    if not evidence_text and not summary_text:
        raise HTTPException(
            400, "At least one of 'evidence' or 'summary' must be provided."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior consultant radiologist. The caller provides a "
                "summary (definitive for pneumonia status) and an evidence block. "
                "Write a concise narrative report in British English (≈150–250 words) "
                "with: opening diagnosis (as per summary), imaging description, "
                "clinical significance, and recommendations. Do not include numbers, "
                "probabilities, or mention AI."
            ),
        },
        {
            "role": "user",
            "content": (
                "### Confirmed summary\n"
                "```text\n"
                f"{summary_text or 'None provided'}\n"
                "```\n\n"
                "### Supporting evidence\n"
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


# ---------- /analyse  (Maverick, descriptive only, no diagnosis) -------------
@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    extra_instructions: str = Form(
        "",
        description="Optional extra guidance (kept non-diagnostic).",
    ),
) -> JSONResponse:
    """
    Non-diagnostic descriptive analysis of a chest X-ray using Groq Maverick.
    Returns structured JSON. Mentions orientation markers and precise lung
    locations where applicable. No diagnosis or clinical advice.
    """
    try:
        img_bytes: bytes = await file.read()
        Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}") from exc

    data_url = make_data_url_under_limit(img_bytes, file.filename or "upload.png")

    json_request_instructions = f"""
You are an expert radiography describer. Produce a purely descriptive account
in British English of the chest X-ray shown. Do not provide a diagnosis,
probability, recommendation, or clinical certainty language. Focus strictly on
what is visible. When relevant, name precise lung locations (e.g., right upper
zone, left lower zone, perihilar region, costophrenic angle).

Return a strict JSON object with exactly these keys:

{{
  "image_orientation_marker": "One of: 'R', 'L', 'Both', 'None visible', or 'Unclear'.",
  "view_and_positioning": "Short text (e.g., PA/AP/Supine/Erect if inferable; otherwise 'Unclear').",
  "exposure_contrast": "Short text on exposure/contrast (e.g., under-/over-exposed, adequate), citing where applicable.",
  "anatomical_description": "Short paragraph describing visible anatomy and lung fields, mentioning specific lung locations when appropriate.",
  "devices_and_artefacts": ["List devices/artefacts/implants/foreign bodies if seen; otherwise empty array."],
  "suspected_structural_changes": "Short text noting any visual discontinuities/irregularities/fracture-like lines or asymmetries WITHOUT naming a diagnosis; include locations if relevant.",
  "overall_description": "2–3 sentences giving a plain-English, non-diagnostic overview of what the image depicts."
}}
{extra_instructions.strip()}
""".strip()

 messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": json_request_instructions},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
]

    raw = call_groq(
        messages,
        model=GROQ_VISION_MODEL_MAVERICK,
        temperature=0.2,
        top_p=1,
        response_format={"type": "json_object"},
        max_completion_tokens=1024,
    )

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    return JSONResponse(
        {
            "model": GROQ_VISION_MODEL_MAVERICK,
            "analysis": parsed if isinstance(parsed, dict) else None,
            "raw": raw if not isinstance(parsed, dict) else None,
            "note": "Descriptive only. No diagnostic interpretation provided.",
        }
    )
