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
    label_text:  str                  # e.g. "This image demonstrates pneumonia."
    buckets:     Dict[str, float]
    votes:       List[Dict[str, Any]]

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
    header = payload.label_text
    if payload.final_label == "pneumonia":
        instruction = (
            "Now describe the anatomical location of the pneumonia "
            "and any key radiological findings."
        )
    elif payload.final_label == "no_evidence":
        instruction = (
            "Now comment on the lung fields, mediastinum, "
            "and overall radiographic quality."
        )
    else:
        instruction = (
            "Now recommend appropriate next steps, such as further imaging "
            "or clinical correlation."
        )
    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Assessment Summary: {header}\n\n"
        "Supporting data:\n"
        f"{json.dumps({'buckets': payload.buckets, 'votes': payload.votes}, indent=2)}\n\n"
        f"Instruction: {instruction}"
    )
    answer = call_groq(prompt)
    return JSONResponse({"answer": answer, **payload.dict()})


@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    final_label: str        = Form(...),
    label_text:  str        = Form(...),
    buckets:      str       = Form(...),
    votes:        str       = Form(...),
):
    # Parse JSON-encoded buckets and votes
    try:
        buckets_obj = json.loads(buckets)
        votes_obj   = json.loads(votes)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in form fields 'buckets' or 'votes'.")

    header = label_text
    if final_label == "pneumonia":
        detail = (
            "Draft 5–7 bullet points focusing on the presence of pneumonia, "
            "including anatomical location, salient findings, and recommendations."
        )
    elif final_label == "no_evidence":
        detail = (
            "Draft 5–7 bullet points describing the chest X-ray, "
            "noting pulmonary fields, mediastinum, and absence of pneumonia."
        )
    else:
        detail = (
            "Draft 5–7 bullet points summarising the uncertainty, key observations, "
            "and recommended next steps."
        )

    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Assessment Summary: {header}\n\n"
        "Supporting data:\n"
        f"{json.dumps({'buckets': buckets_obj, 'votes': votes_obj}, indent=2)}\n\n"
        f"{detail}"
    )
    report_text = call_groq(prompt)
    return JSONResponse({
        "report": report_text,
        "final_label": final_label,
        "label_text": label_text,
        "buckets": buckets_obj,
        "votes": votes_obj,
    })


@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn):
    if not payload.evidence.strip():
        raise HTTPException(400, "Field 'evidence' is empty.")
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
