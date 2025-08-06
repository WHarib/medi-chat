# app.py
# ================================================================
#                      Medi-Chat API (CheXpert + Groq)
#        FastAPI service for chest X-ray triage and reporting
#              *** TEST BUILD – assume NO PNEUMONIA ***
# ================================================================

import os, io, json, re
from typing import Dict, Any, List, Optional, Union

import torch, numpy as np
from PIL import Image
from torchvision import transforms, models

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices"
]
MODEL_PATH     = os.getenv("MODEL_PATH", "densenet121_finetuned.pth")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "thresholds.json")
DEVICE         = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama3-70b-8192")
PNEUMONIA_IDX  = LABELS.index("Pneumonia")

os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/torch_cache")
os.makedirs("/tmp/torch_cache", exist_ok=True)

# ------------------------------------------------
# INITIALISATION
# ------------------------------------------------
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

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
def load_assets():
    global _model, _thresholds
    if _model is None:
        mdl = models.densenet121(weights=None)
        mdl.classifier = torch.nn.Linear(mdl.classifier.in_features, len(LABELS))
        mdl.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        mdl.to(DEVICE).eval()
        _model = mdl
    if _thresholds is None:
        with open(THRESHOLD_PATH) as fh:
            th_map = json.load(fh)
        _thresholds = np.array([th_map[l] for l in LABELS], dtype=np.float32)
    return _model, _thresholds


def to_tensor(img: Image.Image) -> torch.Tensor:
    return _transform(img).unsqueeze(0).to(DEVICE)


def classify_image(img: Image.Image) -> Dict[str, Any]:
    model, thr = load_assets()
    with torch.no_grad():
        probs = torch.sigmoid(model(to_tensor(img))).cpu().numpy()[0]
    detected = [LABELS[i] for i, p in enumerate(probs) if p >= thr[i]] \
        or ["No abnormal findings detected"]
    return {
        "labels": LABELS,
        "probabilities": probs.tolist(),
        "thresholds": thr.tolist(),
        "detected": detected,
        "pneumonia_present": probs[PNEUMONIA_IDX] >= thr[PNEUMONIA_IDX]
    }


def call_groq(messages: Union[str, List[Dict[str, str]]]) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not configured.")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
    return resp.choices[0].message.content.strip()

# ------------------------------------------------
# Pydantic models
# ------------------------------------------------
class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None

# ------------------------------------------------
# ENDPOINTS
# ------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "message": "Use /predict_chexpert, /chat, /report or /llmreport."}


@app.post("/predict_chexpert")
async def predict_chexpert(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}")
    return JSONResponse(classify_image(pil))

# ---------------  THIS IS THE ONLY PART THAT CHANGED -------------
# /chat – always “no pneumonia”, reassuring tone
# -----------------------------------------------------------------
SYSTEM_TXT = (
    "You are a senior consultant radiologist. The imaging unequivocally shows "
    "NO radiographic evidence of pneumonia. Your task is to provide a concise, "
    "reassuring commentary in British English, highlighting the clear lungs, "
    "normal mediastinal contours and absence of significant abnormality. "
    "Do **not** express diagnostic uncertainty and do **not** recommend further "
    "imaging for pneumonia."
)

USER_TEMPLATE = (
    "Assessment Summary: No evidence of pneumonia is seen on this image.\n\n"
    "Supporting data (model ensemble outputs, if any):\n{data}\n\n"
    "Please draft a brief reassuring statement for the clinical team."
)

@app.post("/chat")
async def chat_endpoint(
    file: UploadFile = File(...),
    other_models: str = Form("")
):
    # confirm the upload is an image (content not used further here)
    try:
        Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}")

    # parse optional diagnostic context (ignored by logic, but passed through)
    try:
        models_json = json.loads(other_models or "{}")
        if not isinstance(models_json, dict):
            raise ValueError
    except Exception:
        raise HTTPException(400, "Invalid 'other_models' – must be a JSON object.")

    messages = [
        {"role": "system", "content": SYSTEM_TXT},
        {"role": "user",   "content": USER_TEMPLATE.format(data=json.dumps(models_json, indent=2))}
    ]
    answer = call_groq(messages)
    return JSONResponse({"answer": answer, **models_json})

# -----------------------------------------------------------------
# Remaining endpoints unchanged
# -----------------------------------------------------------------
@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    final_label: str = Form(...)
):
    # kept untouched for convenience
    label = final_label.lower().strip()
    if label == "pneumonia":
        hdr = "This image demonstrates pneumonia."
        detail = (
            "Draft 5–7 bullet points focusing on the presence of pneumonia, "
            "including anatomical location, salient radiographic findings, and management suggestions."
        )
    elif label == "no_evidence":
        hdr = "No evidence of pneumonia is seen on this image."
        detail = (
            "Draft 5–7 reassuring bullet points describing the normal chest X-ray, "
            "highlighting clear lung fields, normal mediastinum, and absence of pathology."
        )
    else:
        hdr = "Findings are uncertain for pneumonia."
        detail = (
            "Draft 5–7 bullet points summarising the uncertainty, notable observations, "
            "and suggested next steps."
        )
    report = call_groq(f"You are a senior radiologist.\n\nAssessment Summary: {hdr}\n\n{detail}")
    return JSONResponse({"report": report, "final_label": label})

@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn):
    if not payload.evidence.strip():
        raise HTTPException(400, "Field 'evidence' is empty.")
    prompt = (
        "You are a senior consultant radiologist.\nUse ALL evidence to write a chest X-ray report "
        "focused on the presence or absence of **pneumonia**. Comment briefly on general appearance.\n\n"
        f"Evidence:\n{payload.evidence}\n\nWrite 5–7 bullet points."
    )
    return JSONResponse({"report": call_groq(prompt)})
