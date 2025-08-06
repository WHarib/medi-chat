# app.py
# ================================================================
#                    Medi-Chat API (CheXpert + Groq)
# ================================================================

import os, io, json, re
from typing import Dict, Any, List, Optional, Union

import torch, numpy as np
from PIL import Image
from torchvision import transforms, models
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel

# ------------------------------------------------
# CONFIG (unchanged)
# ------------------------------------------------
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices"
]
MODEL_PATH     = os.getenv("MODEL_PATH", "densenet121_finetuned.pth")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "thresholds.json")
DEVICE         = ("cuda" if torch.cuda.is_available()
                  else ("mps" if torch.backends.mps.is_available() else "cpu"))
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama3-70b-8192")
PNEUMONIA_IDX  = LABELS.index("Pneumonia")

os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/torch_cache")
os.makedirs("/tmp/torch_cache", exist_ok=True)

# ------------------------------------------------
# FASTAPI SETUP (unchanged)
# ------------------------------------------------
app = FastAPI(title="Medi-Chat API (CheXpert + Groq)",
              docs_url="/docs", redoc_url="/redoc")

_model: Optional[torch.nn.Module] = None
_thresholds: Optional[np.ndarray] = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------------------------
# UTILITIES (unchanged)
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
    pneu_prob = float(probs[PNEUMONIA_IDX])
    pneu_thr  = float(thr[PNEUMONIA_IDX])
    pneu_flag = bool(pneu_prob >= pneu_thr)
    return {
        "labels": LABELS,
        "probabilities": probs.tolist(),
        "thresholds": thr.tolist(),
        "detected": detected,
        "pneumonia_probability": pneu_prob,
        "pneumonia_threshold": pneu_thr,
        "pneumonia_present": pneu_flag
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
# Pydantic model (unchanged)
# ------------------------------------------------
class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None

# ------------------------------------------------
# ENDPOINTS
# ------------------------------------------------
@app.get("/")
def root():
    return {"ok": True,
            "message": "Use /predict_chexpert, /chat, /report or /llmreport."}

@app.post("/predict_chexpert")
async def predict_chexpert(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return JSONResponse(classify_image(pil))
    except Exception as exc:
        raise HTTPException(500, f"Internal error in /predict_chexpert: {exc!r}")

# ---------- DEBUG-ENABLED /chat --------------------------------------------
LABEL_SET = {"pneumonia", "no_evidence", "unsure"}
SYS_TEMPLATES = {
    "pneumonia":
        "You are a senior consultant radiologist. The image shows pneumonia. "
        "Provide a confident description of its anatomical location, radiographic "
        "features and likely severity.",
    "no_evidence":
        "You are a senior consultant radiologist. The image shows **no evidence "
        "of pneumonia**. Provide a concise, reassuring statement in British "
        "English, highlighting clear lungs and normal mediastinal contours. "
        "Do **not** express diagnostic uncertainty or recommend further imaging "
        "for pneumonia.",
    "unsure":
        "You are a senior consultant radiologist. Findings are equivocal for "
        "pneumonia. Briefly state the uncertainty and outline sensible next "
        "steps (e.g. clinical correlation, follow-up imaging)."
}
USER_TEMPLATES = {
    "pneumonia":
        "Assessment summary: This image demonstrates pneumonia.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Describe location and features.",
    "no_evidence":
        "Assessment summary: No evidence of pneumonia is seen on this image.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Reassure; do not express uncertainty.",
    "unsure":
        "Assessment summary: Findings are uncertain for pneumonia.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Suggest next steps."
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
    final_label: str = Form(""),
    other_models: str = Form("")
):
    # 1) Validate image
    try:
        Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    # 2) Gather form keys & raw candidate strings
    form = await request.form()
    form_keys = list(form.keys())

    raw_candidates = [
        final_label,
        form.get("json", ""),
        other_models,
        form.get("other_models", "")
    ]
    # extract from JSON blobs in any of those fields
    for blob in [other_models, form.get("json", ""), form.get("other_models", "")]:
        raw_candidates += labels_from_any_json(blob or "")

    # 3) Detect which label we’ll use
    chosen_label = "unsure"
    matched_string = ""
    for candidate in raw_candidates:
        lbl = detect_label(candidate)
        if lbl:
            chosen_label = lbl
            matched_string = candidate
            break

    # 4) Build context dict for prompt
    try:
        ctx = json.loads(other_models or form.get("json", "") or "{}")
        if not isinstance(ctx, dict):
            ctx = {}
    except:
        ctx = {}

    # 5) Call the LLM
    messages = [
        {"role": "system",  "content": SYS_TEMPLATES[chosen_label]},
        {"role": "user",    "content": USER_TEMPLATES[chosen_label].format(
            data=json.dumps(ctx, indent=2))}
    ]
    answer = call_groq(messages)

    # 6) Return debug info + answer
    return JSONResponse({
        "answer":         answer,
        "detected_label": chosen_label,
        "matched_string": matched_string,
        "raw_candidates": raw_candidates,
        "form_keys":      form_keys,
        "final_label":    chosen_label,
        **ctx
    })

# ---------- /report & /llmreport remain unchanged ---------------------------
@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    final_label: str = Form(...)
):
    lbl = detect_label(final_label)
    if lbl == "":
        raise HTTPException(400, "final_label must be pneumonia, no_evidence or unsure.")
    hdr_map = {
        "pneumonia":   "This image demonstrates pneumonia.",
        "no_evidence": "No evidence of pneumonia is seen on this image.",
        "unsure":      "Findings are uncertain for pneumonia."
    }
    detail_map = {
        "pneumonia":   "Draft 5–7 bullet points on pneumonia presence & features.",
        "no_evidence": "Draft 5–7 reassuring bullet points on normal CXR; no uncertainty.",
        "unsure":      "Draft 5–7 bullet points on uncertainty & next steps."
    }
    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Assessment Summary: {hdr_map[lbl]}\n\n"
        f"{detail_map[lbl]}"
    )
    return JSONResponse({"report": call_groq(prompt), "final_label": lbl})

@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn):
    if not payload.evidence.strip():
        raise HTTPException(400, "Field 'evidence' is empty.")
    PROMPT = (
        "You are a senior consultant radiologist.\n"
        "Using **all** the evidence below, write exactly **5–7 bullet points** "
        "focused on pneumonia presence/absence & general appearance.\n\n"
        "Your entire output **must** be bullet points only, each starting with '-'.\n\n"
        f"Evidence:\n{payload.evidence}\n\n"
        "Begin bullet list:"
    )
    return JSONResponse({"report": call_groq(PROMPT)})
