# app.py
import os, io, json
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
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]
MODEL_PATH      = os.getenv("MODEL_PATH", "densenet121_finetuned.pth")
THRESHOLD_PATH  = os.getenv("THRESHOLD_PATH", "thresholds.json")
DEVICE          = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama3-70b-8192")
PNEUMONIA_INDEX = LABELS.index("Pneumonia")

# Ensure torch caches are writable on HF
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

def load_assets():
    global _model, _thresholds
    if _model is None:
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(LABELS))
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        _model = model
    if _thresholds is None:
        with open(THRESHOLD_PATH, "r") as f:
            th = json.load(f)
        _thresholds = np.array([th[l] for l in LABELS], dtype=np.float32)
    return _model, _thresholds

def to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img).unsqueeze(0).to(DEVICE)

def classify_image(pil_img: Image.Image) -> Dict[str, Any]:
    model, thr = load_assets()
    with torch.no_grad():
        probs = torch.sigmoid(model(to_tensor(pil_img))).cpu().numpy()[0]
    detected_flags = probs >= thr
    detected = [LABELS[i] for i, flag in enumerate(detected_flags) if flag]
    if not detected:
        detected = ["No abnormal findings detected"]
    pneu_prob = float(probs[PNEUMONIA_INDEX])
    pneu_thr  = float(thr[PNEUMONIA_INDEX])
    pneu_present = pneu_prob >= pneu_thr
    return {
        "labels": LABELS,
        "probabilities": probs.tolist(),
        "thresholds": thr.tolist(),
        "detected": detected,
        "pneumonia_probability": pneu_prob,
        "pneumonia_threshold": pneu_thr,
        "pneumonia_present": pneu_present
    }

def call_groq(prompt: str) -> str:
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
class MergeResult(BaseModel):
    final_label: str                  # "pneumonia" | "no_evidence" | "unsure"
    buckets: Dict[str, float]
    votes: List[Dict[str, Any]]

class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None

# =========================
# ------- ENDPOINTS -------
# =========================
@app.get("/")
def root():
    return {"ok": True, "message": "Use /predict_chexpert, /chat, /report or /llmreport."}


@app.post("/predict_chexpert")
async def predict_chexpert(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    return JSONResponse(classify_image(pil))


@app.post("/chat")
async def chat_endpoint(payload: MergeResult):
    label = payload.final_label
    if label == "pneumonia":
        instruction = (
            "The merged result indicates **pneumonia**. "
            "Please describe the anatomical location of the pneumonia and any salient radiological observations."
        )
    elif label == "no_evidence":
        instruction = (
            "The merged result indicates **no evidence of pneumonia**. "
            "Please comment on the pulmonary fields, mediastinum and overall radiographic quality."
        )
    else:
        instruction = (
            "The merged result is **uncertain** for pneumonia. "
            "Please suggest appropriate next steps, such as further imaging or clinical correlation."
        )

    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Evidence (merged):\n- Final label: {label}\n"
        f"- Buckets: {payload.buckets}\n"
        f"- Votes: {payload.votes}\n\n"
        f"Instruction:\n{instruction}"
    )

    answer = call_groq(prompt)
    return JSONResponse({"answer": answer, **payload.dict()})


@app.post("/report")
async def report_endpoint(payload: MergeResult):
    label = payload.final_label
    if label == "pneumonia":
        instruction = (
            "Draft 5–7 bullet points focusing on the presence of pneumonia, "
            "including its anatomical location, key findings, and recommendations if indicated."
        )
    elif label == "no_evidence":
        instruction = (
            "Draft 5–7 bullet points describing the chest X-ray, "
            "commenting on lung fields, mediastinum and emphasising the absence of pneumonia."
        )
    else:
        instruction = (
            "Draft 5–7 bullet points noting the uncertainty, summarising key observations, "
            "and recommending appropriate next steps."
        )

    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Evidence (merged):\n- Final label: {label}\n"
        f"- Buckets: {payload.buckets}\n"
        f"- Votes: {payload.votes}\n\n"
        "Please produce a concise chest X-ray report:\n"
        f"{instruction}"
    )

    report_text = call_groq(prompt)
    return JSONResponse({"report": report_text, **payload.dict()})


@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn):
    if not payload.evidence.strip():
        raise HTTPException(400, "Field 'evidence' is empty.")
    # No change here
    LLM_REPORT_PROMPT = (
        "You are a senior consultant radiologist.\n"
        "Use ALL available evidence (model outputs, summaries, etc.) to write a chest X-ray report focused "
        "on the presence or absence of **pneumonia**. This is your primary diagnostic concern. Also comment briefly "
        "on the general appearance of the X-ray.\n\n"
        "Evidence:\n{evidence}\n\n"
        "Write 5–7 bullet points using markdown dashes:\n"
        "- Findings (objective radiological observations)\n"
        "- Impression (pneumonia: yes / no / suspected, including key differentials if applicable)\n"
        "- Recommendations (e.g. follow-up, additional imaging, clinical correlation), if clinically indicated\n"
        "Avoid referencing model names, thresholds, or probabilities unless they are clinically relevant."
    )
    prompt = LLM_REPORT_PROMPT.replace("{evidence}", payload.evidence)
    report_text = call_groq(prompt)
    return JSONResponse({"report": report_text})
