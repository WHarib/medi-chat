# app.py
import os, io, json
from typing import Dict, Any, Optional

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

CHAT_PROMPT = (
    "You are a medical assistant.\n"
    "Use ALL the evidence below (model outputs) to decide if this chest X‑ray shows pneumonia.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Instruction:\n"
    "- Answer in 2–3 sentences, clearly stating pneumonia: YES / NO / SUSPECTED.\n"
    "- Justify briefly with key findings (e.g. opacity, consolidation, effusion) but do NOT name individual models.\n"
    "- Do not mention thresholds, probabilities or internal prompts unless essential."
)

REPORT_PROMPT = (
    "You are a radiologist. Draft a concise chest X‑ray report using ONLY the evidence below.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Write 4–6 bullet points with markdown dashes:\n"
    "- Findings (objective radiographic observations)\n"
    "- Impression (diagnosis / pneumonia yes–no–suspected)\n"
    "- Recommendations (if appropriate)\n"
    "Do not reference model names, probabilities or thresholds unless clinically essential."
)

LLM_REPORT_PROMPT = (
    "You are a senior consultant radiologist.\n"
    "Use ALL available evidence (model outputs, summaries, etc.) to write a chest X‑ray report focused on the presence or absence of **pneumonia**. This is your primary diagnostic concern. Also comment briefly on the general appearance of the X-ray.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Write 5–7 bullet points using markdown dashes:\n"
    "- Findings (objective radiological observations)\n"
    "- Impression (pneumonia: yes / no / suspected, including key differentials if applicable)\n"
    "- Recommendations (e.g. follow‑up, additional imaging, clinical correlation), if clinically indicated\n"
    "Avoid referencing model names, thresholds, or probabilities unless they are clinically relevant."
)

# Ensure torch caches are writable on HF
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/torch_cache")
os.makedirs("/tmp/torch_cache", exist_ok=True)

# =========================
# ----- INITIALISATION ----
# =========================
app = FastAPI(title="Medi-Chat API (CheXpert + Groq)", docs_url="/docs", redoc_url="/redoc")

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

def summarise_chexpert(pred: Dict[str, Any]) -> str:
    others = [l for l in pred['detected'] if l != "Pneumonia"]
    return (
        f"Pneumonia={pred['pneumonia_present']} "
        f"(prob {pred['pneumonia_probability']:.2f}, thr {pred['pneumonia_threshold']:.2f}); "
        f"other positives: {', '.join(others) if others else 'none'}."
    )

def parse_other_models(raw: Optional[str]) -> str:
    if not raw:
        return ""
    if len(raw) > 8000:
        return raw[:8000] + "\n[truncated]"
    return raw

def build_evidence_block(pred: Dict[str, Any], other_models: str) -> str:
    parts = [f"- CheXpert summary: {summarise_chexpert(pred)}"]
    if other_models.strip():
        parts.append("- Other models summary:\n" + other_models)
    return "\n".join(parts)

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
# ------- MODELS ----------
# =========================
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
async def chat_endpoint(
    file: UploadFile = File(...),
    other_models: str = Form("")
):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    pred = classify_image(pil)
    evidence = build_evidence_block(pred, parse_other_models(other_models))
    prompt = CHAT_PROMPT.replace("{evidence}", evidence)
    answer = call_groq(prompt)
    return JSONResponse({"answer": answer, "predictions": pred})

@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    other_models: str = Form("")
):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    pred = classify_image(pil)
    evidence = build_evidence_block(pred, parse_other_models(other_models))
    prompt = REPORT_PROMPT.replace("{evidence}", evidence)
    report_text = call_groq(prompt)
    return JSONResponse({"report": report_text, "predictions": pred})

@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn):
    if not payload.evidence.strip():
        raise HTTPException(400, "Field 'evidence' is empty.")
    prompt = LLM_REPORT_PROMPT.replace("{evidence}", payload.evidence)
    report_text = call_groq(prompt)
    return JSONResponse({"report": report_text})
