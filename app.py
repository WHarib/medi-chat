import os, io, json, base64
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from groq import Groq

# ---------------- CONFIG ----------------
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]
MODEL_PATH = os.getenv("MODEL_PATH", "densenet121_finetuned.pth")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "thresholds.json")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
GROQ_KEY = os.getenv("GROQ_API_KEY")  # set this in HF Secrets

# Grad-CAM?
ENABLE_GRADCAM = False  # flip to True if you want heatmaps; add code accordingly
# ---------------------------------------

app = FastAPI(title="Medi-Chat API (CheXpert + Groq)")

# ---- Load model & thresholds ----
_model = None
_thresholds = None
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_assets():
    global _model, _thresholds
    if _model is None:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(LABELS))
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        _model = model
    if _thresholds is None:
        with open(THRESHOLD_PATH, "r") as f:
            tdict = json.load(f)
        _thresholds = np.array([tdict[l] for l in LABELS], dtype=np.float32)
    return _model, _thresholds

def to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img).unsqueeze(0).to(DEVICE)

def predict_chexpert(pil_img: Image.Image) -> Dict[str, Any]:
    model, thr = load_assets()
    with torch.no_grad():
        logits = model(to_tensor(pil_img))
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    detected = probs >= thr
    out = {
        "labels": LABELS,
        "probabilities": probs.tolist(),
        "thresholds": thr.tolist(),
        "detected": [LABELS[i] for i, flag in enumerate(detected) if flag],
    }
    # If nothing detected, keep "No Finding"
    if not out["detected"]:
        out["detected"] = ["No abnormal findings detected"]
    return out

def groq_answer(context_labels: List[str], question: str) -> str:
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY not configured")
    client = Groq(api_key=GROQ_KEY)
    context = f"The following CheXpert conditions were predicted from the chest X-ray: {', '.join(context_labels)}."
    prompt = f"{context}\n\nPatient question: {question}\n\nGive a short, clear medical answer (2–3 sentences only). Avoid long explanations."
    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# ------------- Endpoints -------------

@app.get("/")
def root():
    return {"ok": True, "message": "Use /predict_chexpert or /chat"}

@app.post("/predict_chexpert")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    result = predict_chexpert(pil)
    return JSONResponse(result)

@app.post("/chat")
async def chat_endpoint(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    pred = predict_chexpert(pil)
    answer = groq_answer(pred["detected"], question)
    return JSONResponse({
        "answer": answer,
        "predictions": pred
    })
