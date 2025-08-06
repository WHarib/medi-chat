# app.py
# ================================================================
#                    Medi-Chat API (CheXpert + Groq)
# ================================================================

import os, io, json, re, base64
from typing import Dict, Any, List, Optional, Union

import torch, numpy as np
from PIL import Image
from torchvision import transforms, models
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
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
DEVICE         = ("cuda" if torch.cuda.is_available()
                  else ("mps" if torch.backends.mps.is_available() else "cpu"))
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama3-70b-8192")   # default text model
PNEUMONIA_IDX  = LABELS.index("Pneumonia")

os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/torch_cache")
os.makedirs("/tmp/torch_cache", exist_ok=True)

# ------------------------------------------------
# FASTAPI
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


def call_groq(
    messages: Union[str, List[Dict[str, str]]],
    model: str = GROQ_MODEL,
    **kwargs
) -> str:
    """Wrapper around Groq chat-completion with optional model override."""
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not configured.")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return resp.choices[0].message.content.strip()

# ------------------------------------------------
# Pydantic model
# ------------------------------------------------
class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None

# ------------------------------------------------
# ENDPOINTS  (predict_chexpert, chat, report unchanged)
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

# ---------- /chat (unchanged) -----------------------------------------------
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
        "Do **not** express diagnostic uncertainty or recommend further imaging.",
    "unsure":
        "You are a senior consultant radiologist. Findings are equivocal for "
        "pneumonia. Briefly state the uncertainty and outline sensible next steps."
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
    try:
        Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}")

    form = await request.form()
    raw_candidates = [
        final_label,
        form.get("json", ""),
        other_models,
        form.get("other_models", "")
    ]
    for blob in (other_models, form.get("json", ""), form.get("other_models", "")):
        raw_candidates += labels_from_any_json(blob or "")

    chosen_label = next(
        (detect_label(c) for c in raw_candidates if detect_label(c)), "unsure"
    )
    if chosen_label == "unsure" and all(not c.strip() for c in raw_candidates):
        chosen_label = "no_evidence"

    try:
        ctx = json.loads(other_models or form.get("json", "") or "{}")
        if not isinstance(ctx, dict):
            ctx = {}
    except Exception:
        ctx = {}

    messages = [
        {"role": "system", "content": SYS_TEMPLATES[chosen_label]},
        {"role": "user",
         "content": USER_TEMPLATES[chosen_label].format(
             data=json.dumps(ctx, indent=2))}
    ]
    answer = call_groq(messages)                 # default text model
    return JSONResponse({"answer": answer})

# ---------- /report (unchanged) ---------------------------------------------
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
    report = call_groq(prompt)   # default text model
    return JSONResponse({"report": report, "final_label": lbl})

# ---------- /llmreport  (pass-through version) ------------------------------
@app.post("/llmreport")
async def llmreport_endpoint(request: Request):
    """
    Accepts either JSON or form-data.
    • `evidence`  (or `evidences`) – raw probabilities / model output
    • `summary`   – confirmed narrative from previous step

    Both fields are forwarded **as-is** to GPT-OSS-120B.
    The model is instructed that the summary is definitive regarding pneumonia.
    """
    # ---------------- read body flexibly -----------------------------------
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        evidence_text = body.get("evidence") or body.get("evidences", "")
        summary_text  = body.get("summary", "")
    else:
        form = await request.form()
        evidence_text = form.get("evidence") or form.get("evidences", "")
        summary_text  = form.get("summary", "")

    evidence_text = str(evidence_text).strip()
    summary_text  = str(summary_text).strip()

    if not evidence_text and not summary_text:
        raise HTTPException(
            400,
            "Provide at least one of these fields: evidence / summary."
        )

    # ---------------- build prompt ----------------------------------------
    prompt = (
        "You are a senior consultant radiologist.\n\n"
        "**Clinical summary (confirmed, 100 % reliable):**\n"
        f"{summary_text or '*No summary provided.*'}\n\n"
        "**Additional model evidence (probabilities, free text, etc.):**\n"
        f"{evidence_text or '*No evidence provided.*'}\n\n"
        "Using ALL of the above, write **exactly 5–7 bullet points** "
        "in British English:\n"
        " - Begin with the confirmed pneumonia conclusion from the summary.\n"
        " - Then comment on any other findings suggested by the probabilities.\n"
        "Do NOT contradict the summary regarding pneumonia, and do NOT add "
        "headings, dates or patient identifiers.\n\n"
        "Begin bullet list:"
    )

    # ---------------- call Groq -------------------------------------------
    report = call_groq(
        prompt,
        model="openai/gpt-oss-120b",
        max_completion_tokens=8192,
        temperature=0.7,
        top_p=1,
        reasoning_effort="medium"
    )
    return JSONResponse({"report": report})
# -----------------------------------------------------------------
#  NEW ENDPOINT: /vision_report  – GPT-4-o (120 B) image analysis
# -----------------------------------------------------------------
import base64

@app.post("/vision_report")
async def vision_report(
    file: UploadFile = File(...),
    extra_prompt: str = Form("", description="Optional extra instructions")
):
    """
    Feed an X-ray (or any image) to *openai/gpt-oss-120b* via Groq.

    The model is asked to:
      1. Describe the image objectively.
      2. Comment on any abnormalities (or state that none are seen).
      3. Provide a concise summary suitable for a clinical note.

    Returns the model's plain-text answer.
    """
    # 1) Read and sanity-check image
    try:
        img_bytes = await file.read()
        Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}")

    # 2) Convert to data-URL
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else "png"
    data_url = (
        f"data:image/{ext};base64," + base64.b64encode(img_bytes).decode()
    )

    # 3) Build multimodal message payload
    vision_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please perform **three tasks** on the chest X-ray below:\n"
                        "1. **Objective description** of visible anatomy and features.\n"
                        "2. **Comment on abnormalities** (state 'None seen' if normal).\n"
                        "3. **Concise overall summary** (like senior radiologist).\n\n"
                        f"{extra_prompt.strip()}"
                    ).strip()
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }
            ]
        }
    ]

    # 4) Call Groq (GPT-4-o 120 B)
    answer = call_groq(
        vision_messages,
        model="openai/gpt-oss-120b",
        max_completion_tokens=1024,
        temperature=0.3,
        top_p=1,
        reasoning_effort="medium"
    )

    return JSONResponse({"report": answer})


