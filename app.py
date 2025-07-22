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

# =========================
# -------- CONFIG ---------
# =========================
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

MODEL_PATH       = os.getenv("MODEL_PATH", "densenet121_finetuned.pth")
THRESHOLD_PATH   = os.getenv("THRESHOLD_PATH", "thresholds.json")
DEVICE           = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")  # only needed for /chat and /report
PNEUMONIA_INDEX  = LABELS.index("Pneumonia")

# Hard-coded prompts
CHAT_PROMPT = (
    "You are a medical assistant. Decide if the chest X‑ray shows pneumonia.\n\n"
    "CheXpert model: {chexpert_summary}\n"
    "Other model outputs: {other_models_summary}\n\n"
    "Task: In 2–3 sentences, state clearly whether pneumonia is present or not, "
    "justify briefly using the evidence above, and mention any other critical findings."
)

REPORT_PROMPT = (
    "You are a radiologist. Write a concise bullet-point report for this chest X‑ray.\n\n"
    "CheXpert model: {chexpert_summary}\n"
    "Other model outputs: {other_models_summary}\n\n"
    "Provide 4–6 bullet points covering:\n"
    "- Findings (objective image features)\n"
    "- Impression (diagnosis / pneumonia yes/no)\n"
    "- Recommendations (if appropriate)\n"
    "Keep it factual and do not speculate beyond the evidence provided."
)

# Optional: ensure torch cache is writable (not strictly needed if weights=None)
os.environ.setdefault("TORCH_HOME", "/app/torch_cache")
os.makedirs("/app/torch_cache", exist_ok=True)

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
        # Avoid downloading ImageNet weights
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
    return (
        f"Detected: {', '.join(pred['detected'])}. "
        f"Pneumonia prob: {pred['pneumonia_probability']:.2f}, "
        f"threshold: {pred['pneumonia_threshold']:.2f}, "
        f"present: {pred['pneumonia_present']}"
    )

def parse_other_models(raw: Optional[str]) -> str:
    if not raw:
        return "None provided"
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return "; ".join(f"{k}: {v}" for k, v in data.items())
        if isinstance(data, list):
            return ", ".join(map(str, data))
        return str(data)
    except Exception:
        return raw

def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY is not configured.")
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# =========================
# ------- ENDPOINTS -------
# =========================

@app.get("/")
def root():
    return {"ok": True, "message": "Use /predict_chexpert, /chat or /report."}

@app.post("/predict_chexpert")
async def predict_chexpert(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    result = classify_image(pil)
    return JSONResponse(result)

@app.post("/chat")
async def chat_endpoint(
    file: UploadFile = File(...),
    other_models: str = Form(None)
):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    pred = classify_image(pil)
    chexpert_summary = summarise_chexpert(pred)
    other_models_summary = parse_other_models(other_models)

    prompt = CHAT_PROMPT.format(
        chexpert_summary=chexpert_summary,
        other_models_summary=other_models_summary
    )
    answer = call_groq(prompt)

    return JSONResponse({
        "answer": answer,
        "predictions": pred,
        "other_models_summary": other_models_summary
    })

@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    other_models: str = Form(None)
):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    pred = classify_image(pil)
    chexpert_summary = summarise_chexpert(pred)
    other_models_summary = parse_other_models(other_models)

    prompt = REPORT_PROMPT.format(
        chexpert_summary=chexpert_summary,
        other_models_summary=other_models_summary
    )
    report_text = call_groq(prompt)

    return JSONResponse({
        "report": report_text,
        "predictions": pred,
        "other_models_summary": other_models_summary
    })
