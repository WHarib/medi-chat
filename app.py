# app.py
# ================================================================
#                      Medi-Chat API (CheXpert + Groq)
#           FastAPI service for chest X-ray triage and reporting
# ================================================================

import os
import io
import json
import re
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
    """Lazily load the CNN weights and threshold vector."""
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
    """PIL → normalised batch tensor on the correct device."""
    return _transform(img).unsqueeze(0).to(DEVICE)


def classify_image(img: Image.Image) -> Dict[str, Any]:
    """Run the classifier and package a structured response."""
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
    """Wrapper for the Groq chat-completion API call."""
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
# /chat  – interactive narrative / advice (Groq LLM)
# ----------------------------------------------------------------
NO_PNEUMONIA_SET = {"no_evidence", "no pneumonia", "negative", "clear"}
YES_PNEUMONIA_SET = {"pneumonia", "positive"}
UNCERTAIN_SET = {"unsure", "uncertain", "maybe", "equivocal"}


def normalise_label(raw_label: str) -> str:
    """Lower-case, strip whitespace and collapse repeated spaces."""
    return re.sub(r"\s+", " ", raw_label or "").strip().lower()


def build_prompt(final_label: str,
                 buckets: Dict[str, Any],
                 votes: List[Any]) -> str:
    """Construct a strong deterministic prompt for Groq."""
    if final_label in YES_PNEUMONIA_SET:
        header = "This image demonstrates pneumonia."
        instr = (
            "Describe the anatomical location, radiographic features, and "
            "potential severity."
        )
    elif final_label in NO_PNEUMONIA_SET:
        header = "No evidence of pneumonia is seen on this image."
        instr = (
            "Reassure the clinical team. Comment on clear lung fields, "
            "normal mediastinal contours, and absence of radiographic "
            "abnormalities. **Do not express diagnostic uncertainty.**"
        )
    else:  # uncertain
        header = "Findings are uncertain for pneumonia."
        instr = (
            "Recommend appropriate next steps, such as clinical correlation "
            "or repeat imaging."
        )

    return (
        "You are a senior consultant radiologist.\n\n"
        f"Assessment Summary: {header}\n\n"
        "Supporting data:\n"
        f"{json.dumps({'buckets': buckets, 'votes': votes}, indent=2)}\n\n"
        f"Instruction: {instr}"
    )


@app.post("/chat")
async def chat_endpoint(
    file: UploadFile = File(...),
    other_models: str = Form(""),           # legacy combined JSON result
    final_label: str = Form("")             # preferred explicit field
):
    # ---- image sanity check ------------------------------------------------
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}")

    # ---- parse other_models -------------------------------------------------
    try:
        merged = json.loads(other_models or "{}")
        if not isinstance(merged, dict):
            raise ValueError("Expected a JSON object, got something else.")
    except Exception:
        raise HTTPException(
            400,
            "Invalid 'other_models' format – must be a JSON object."
        )

    # ---- harmonise inputs ---------------------------------------------------
    user_label = normalise_label(final_label)
    merged_label = normalise_label(merged.get("final_label", ""))
    label = user_label or merged_label or "unsure"

    buckets = merged.get("buckets", {})
    votes   = merged.get("votes", [])

    # ---- build prompt & call Groq ------------------------------------------
    prompt  = build_prompt(label, buckets, votes)
    answer  = call_groq(prompt)

    # ---- response -----------------------------------------------------------
    result = {"answer": answer,
              "final_label": label,
              "buckets": buckets,
              "votes": votes}
    return JSONResponse(result)


# ----------------------------------------------------------------
# /report  – template-driven short report (Groq LLM)
# ----------------------------------------------------------------
@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    final_label: str = Form(...)
):
    label_norm = normalise_label(final_label)

    if label_norm in YES_PNEUMONIA_SET:
        label_text = "This image demonstrates pneumonia."
        detail = (
            "Draft 5–7 bullet points focusing on the presence of pneumonia, "
            "including anatomical location, salient radiographic findings, "
            "and management suggestions."
        )
    elif label_norm in NO_PNEUMONIA_SET:
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
        "final_label": label_norm,
        "label_text":  label_text
    })


# ----------------------------------------------------------------
# /llmreport  – evidence-rich bullet report (Groq LLM)
# ----------------------------------------------------------------
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
