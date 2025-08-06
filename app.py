# app.py
# ================================================================
#                      Medi-Chat API (CheXpert + Groq)
#           FastAPI service for chest X-ray triage and reporting
# ================================================================

import os
import io
import json
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel

# =========================
# -------- CONFIG ---------
# =========================
LABELS: List[str] = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices"
]
MODEL_PATH: str      = os.getenv("MODEL_PATH", "densenet121_finetuned.pth")
THRESHOLD_PATH: str  = os.getenv("THRESHOLD_PATH", "thresholds.json")
DEVICE: str          = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
GROQ_API_KEY: str    = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str      = os.getenv("GROQ_MODEL", "llama3-70b-8192")
PNEUMONIA_INDEX: int = LABELS.index("Pneumonia")

os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/torch_cache")
os.makedirs("/tmp/torch_cache", exist_ok=True)

# =========================
# ----- INITIALISATION ----
# =========================
app = FastAPI(
    title="Medi-Chat API (CheXpert + Groq)",
    docs_url="/docs",
    redoc_url="/redoc"
)

_model: Optional[torch.nn.Module] = None
_thresholds: Optional[np.ndarray] = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# ---- UTILITY HELPERS ----
# =========================
def load_assets():
    """Lazy-load CNN weights and threshold vector."""
    global _model, _thresholds
    if _model is None:
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(
            model.classifier.in_features, len(LABELS)
        )
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        _model = model

    if _thresholds is None:
        with open(THRESHOLD_PATH, "r") as fh:
            threshold_map = json.load(fh)
        _thresholds = np.array(
            [threshold_map[label] for label in LABELS], dtype=np.float32
        )

    return _model, _thresholds


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → normalised batch tensor on correct device."""
    return _transform(img).unsqueeze(0).to(DEVICE)


def classify_image(img: Image.Image) -> Dict[str, Any]:
    """Run classifier and package structured response."""
    model, thr = load_assets()
    with torch.no_grad():
        probs = torch.sigmoid(model(to_tensor(img))).cpu().numpy()[0]

    detected_flags = probs >= thr
    detected = [LABELS[i] for i, flag in enumerate(detected_flags) if flag] \
        or ["No abnormal findings detected"]

    pneu_prob = float(probs[PNEUMONIA_INDEX])
    pneu_thr  = float(thr[PNEUMONIA_INDEX])

    return {
        "labels": LABELS,
        "probabilities": probs.tolist(),
        "thresholds": thr.tolist(),
        "detected": detected,
        "pneumonia_probability": pneu_prob,
        "pneumonia_threshold": pneu_thr,
        "pneumonia_present": pneu_prob >= pneu_thr
    }


def call_groq(prompt: str) -> str:
    """Groq chat-completion wrapper."""
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY is not configured.")
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# =========================
# ---- INPUT MODELS -------
# =========================
class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None

# =========================
# ------- ENDPOINTS -------
# =========================
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Use /predict_chexpert, /chat, /report or /llmreport."
    }

@app.post("/predict_chexpert")
async def predict_chexpert(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}")
    return JSONResponse(classify_image(pil))

# ----------------------------------------------------------------
# /chat  – *Hard-wired* “no pneumonia” narrative
# ----------------------------------------------------------------
NO_PNEUMONIA_HEADER = "No evidence of pneumonia is seen on this image."
NO_PNEUMONIA_INSTR  = (
    "Reassure the clinical team. Comment on clear lung fields, normal "
    "mediastinal contours, and absence of radiographic abnormalities. "
    "**Do not express diagnostic uncertainty.**"
)

@app.post("/chat")
async def chat_endpoint(
    file: UploadFile = File(...),
    other_models: str = Form("")
):
    # ---- image sanity check -----------------------------------------------
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}")

    # ---- still parse legacy other_models for completeness ------------------
    try:
        merged = json.loads(other_models or "{}")
        if not isinstance(merged, dict):
            raise ValueError
    except Exception:
        raise HTTPException(
            400,
            "Invalid 'other_models' format – must be a JSON object."
        )

    # ---- build prompt (no branching) --------------------------------------
    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Assessment Summary: {NO_PNEUMONIA_HEADER}\n\n"
        "Supporting data (for context only):\n"
        f"{json.dumps(merged, indent=2)}\n\n"
        f"Instruction: {NO_PNEUMONIA_INSTR}"
    )
    answer = call_groq(prompt)

    # ---- response ----------------------------------------------------------
    return JSONResponse({
        "answer":      answer,
        "final_label": "no_evidence (hard-wired)",
        **merged
    })

# ----------------------------------------------------------------
# The remaining endpoints are unchanged
# ----------------------------------------------------------------
@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    final_label: str = Form(...)
):
    # keep original branching for /report
    if final_label.lower().strip() == "pneumonia":
        label_text = "This image demonstrates pneumonia."
        detail = (
            "Draft 5–7 bullet points focusing on the presence of pneumonia, "
            "including anatomical location, salient radiographic findings, "
            "and management suggestions."
        )
    elif final_label.lower().strip() == "no_evidence":
        label_text = "No evidence of pneumonia is seen on this image."
        detail = (
            "Draft 5–7 reassuring bullet points describing the normal chest "
            "X-ray, highlighting clear lung fields, normal mediastinum, and "
            "absence of pathology. **Do not express diagnostic uncertainty.**"
        )
    else:
        label_text = "Findings are uncertain for pneumonia."
        detail = (
            "Draft 5–7 bullet points summarising the ambiguity, notable "
            "observations, and suggested next steps for clinical evaluation."
        )

    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Assessment Summary: {label_text}\n\n"
        f"{detail}"
    )
    report = call_groq(prompt)
    return JSONResponse({
        "report":      report,
        "final_label": final_label,
        "label_text":  label_text
    })

@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn):
    if not payload.evidence.strip():
        raise HTTPException(400, "Field 'evidence' is empty.")

    PROMPT = (
        "You are a senior consultant radiologist.\n"
        "Use ALL available evidence (model outputs, summaries, etc.) to write "
        "a chest X-ray report focused on the presence or absence of "
        "**pneumonia**. This is your primary diagnostic concern. Also comment "
        "briefly on the general appearance of the X-ray.\n\n"
        f"Evidence:\n{payload.evidence}\n\n"
        "Write 5–7 bullet points using markdown dashes:\n"
        "- Findings (objective radiological observations)\n"
        "- Impression (pneumonia: yes / no / suspected, including key "
        "  differentials if applicable)\n"
        "- Recommendations (e.g. follow-up, additional imaging, "
        "  clinical correlation), if clinically indicated\n"
        "Avoid referencing model names, thresholds, or probabilities unless "
        "they are clinically relevant."
    )
    report_text = call_groq(PROMPT)
    return JSONResponse({"report": report_text})
