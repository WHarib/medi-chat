import os, io, json, base64
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq

MAX_B64_BYTES = 3_600_000   # safety margin under Groq's 4 MB base64 cap
MAX_PIXELS = 33_177_600     # 33 megapixels

def make_data_url_under_limit(img_bytes: bytes, filename: str | None = None) -> str:
    """
    Convert arbitrary input image bytes into a JPEG data URL whose base64 payload
    is best-effort <= MAX_B64_BYTES and <= MAX_PIXELS.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Enforce resolution cap
    if (img.width * img.height) > MAX_PIXELS:
        scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
        img = img.resize((
            max(1, int(img.width * scale)),
            max(1, int(img.height * scale)),
        ))

    # Try qualities at current size
    for quality in (90, 80, 70, 60, 50, 40):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue())
        if len(b64) <= MAX_B64_BYTES:
            return "data:image/jpeg;base64," + b64.decode()

    # If still too big, downscale once and retry
    img = img.resize((
        max(1, int(img.width * 0.8)),
        max(1, int(img.height * 0.8)),
    ))
    for quality in (70, 60, 50, 40, 35):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue())
        if len(b64) <= MAX_B64_BYTES:
            return "data:image/jpeg;base64," + b64.decode()

    # Last resort (may exceed limit, but we tried)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()



# ------------------------------------------------
# CONFIG
# ------------------------------------------------
LABELS: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

MODEL_PATH: str = os.getenv("MODEL_PATH", "densenet121_finetuned.pth")
THRESHOLD_PATH: str = os.getenv("THRESHOLD_PATH", "thresholds.json")
DEVICE: str = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_COMPLETION_TOKENS: int = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "8192"))

# Vision model for /analyse (Maverick)
GROQ_VISION_MODEL_MAVERICK: str = os.getenv(
    "GROQ_VISION_MODEL_MAVERICK",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

PNEUMONIA_IDX: int = LABELS.index("Pneumonia")

os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/torch_cache")
os.makedirs("/tmp/torch_cache", exist_ok=True)

# ------------------------------------------------
# FASTAPI
# ------------------------------------------------
app = FastAPI(
    title="Medi-Chat API (CheXpert + Groq)",
    docs_url="/docs",
    redoc_url="/redoc",
)

_model: Optional[torch.nn.Module] = None
_thresholds: Optional[np.ndarray] = None

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
def load_assets() -> tuple[torch.nn.Module, np.ndarray]:
    global _model, _thresholds
    if _model is None:
        mdl = models.densenet121(weights=None)
        mdl.classifier = torch.nn.Linear(mdl.classifier.in_features, len(LABELS))
        mdl.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        mdl.to(DEVICE).eval()
        _model = mdl

    if _thresholds is None:
        with open(THRESHOLD_PATH, "r") as fh:
            thr_map = json.load(fh)
        _thresholds = np.array([thr_map[lbl] for lbl in LABELS], dtype=np.float32)

    return _model, _thresholds


def to_tensor(img: Image.Image) -> torch.Tensor:
    return _transform(img).unsqueeze(0).to(DEVICE)


def classify_image(img: Image.Image) -> Dict[str, Any]:
    model, thr = load_assets()
    with torch.no_grad():
        probs = torch.sigmoid(model(to_tensor(img))).cpu().numpy()[0]

    detected = [LABELS[i] for i, p in enumerate(probs) if p >= thr[i]] or [
        "No abnormal findings detected"
    ]

    pneu_prob = float(probs[PNEUMONIA_IDX])
    pneu_thr = float(thr[PNEUMONIA_IDX])
    pneu_flag = bool(pneu_prob >= pneu_thr)

    return {
        "labels": LABELS,
        "probabilities": probs.tolist(),
        "thresholds": thr.tolist(),
        "detected": detected,
        "pneumonia_probability": pneu_prob,
        "pneumonia_threshold": pneu_thr,
        "pneumonia_present": pneu_flag,
    }



def call_groq(
    messages: Union[str, Sequence[Dict[str, Any]]],
    *,
    model: str = GROQ_MODEL,
    max_completion_tokens: Optional[int] = None,
    stream: Optional[bool] = None,
    **kwargs: Any,
) -> str:
    """
    Universal wrapper round Groq chat-completion with:
      - streaming support (needed for openai/gpt-oss-120b which often streams),
      - empty-content retry,
      - decommissioned-model fallback.
    """
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY is not configured on the server.")

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    client = Groq(api_key=GROQ_API_KEY)

    # Default to streaming for gpt-oss-120b; otherwise non-streaming
    use_stream = stream if stream is not None else ("gpt-oss-120b" in model)

    def _once(selected_model: str, *, streamed: bool) -> str:
        if streamed:
            chunks: List[str] = []
            completion = client.chat.completions.create(
                model=selected_model,
                messages=list(messages),
                max_completion_tokens=max_completion_tokens or MAX_COMPLETION_TOKENS,
                stream=True,
                **kwargs,
            )
            for chunk in completion:
                try:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        chunks.append(delta.content)
                except Exception:
                    # Ignore malformed chunks
                    pass
            return ("".join(chunks)).strip()
        else:
            resp = client.chat.completions.create(
                model=selected_model,
                messages=list(messages),
                max_completion_tokens=max_completion_tokens or MAX_COMPLETION_TOKENS,
                **kwargs,
            )
            return (resp.choices[0].message.content or "").strip()

    try:
        content = _once(model, streamed=use_stream)
        # If empty, retry once non-streaming, then with a stable fallback
        if not content:
            print(f"[call_groq] Empty content from '{model}' (stream={use_stream}). Retrying non-stream.")
            content = _once(model, streamed=False)
        if not content and model != "llama-3.3-70b-versatile":
            print(f"[call_groq] Still empty; falling back to llama-3.3-70b-versatile.")
            content = _once("llama-3.3-70b-versatile", streamed=False)
        if not content:
            raise HTTPException(502, f"Groq returned empty content for model '{model}'.")
        return content
    except Exception as exc:
        msg = getattr(exc, "message", str(exc))
        if "model_decommissioned" in msg or "has been decommissioned" in msg:
            print(f"[call_groq] Model '{model}' decommissioned. Fallback: llama-3.3-70b-versatile.")
            try:
                return _once("llama-3.3-70b-versatile", streamed=False)
            except Exception as exc2:
                status = getattr(exc2, "status_code", 502)
                detail = getattr(exc2, "message", str(exc2))
                raise HTTPException(status, f"Groq fallback failed: {detail}") from exc2
        status = getattr(exc, "status_code", 502)
        detail = getattr(exc, "message", str(exc))
        raise HTTPException(status, f"Groq request failed: {detail}") from exc


# ------------------------------------------------
# Pydantic model
# ------------------------------------------------
class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None


# ------------------------------------------------
# ENDPOINTS
# ------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "message": "Use /predict_chexpert, /chat, /report, /llmreport, /analyse, or /diagnose.",
    }


# ---------- /predict_chexpert ------------------------------------------------
@app.post("/predict_chexpert")
async def predict_chexpert(file: UploadFile = File(...)) -> JSONResponse:
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return JSONResponse(classify_image(pil))
    except Exception as exc:
        raise HTTPException(500, f"Internal error in /predict_chexpert: {exc!r}") from exc


# ---------- /chat ------------------------------------------------------------
LABEL_SET = {"pneumonia", "no_evidence", "unsure"}
SYS_TEMPLATES = {
    "pneumonia": (
        "You are a senior consultant radiologist. The image shows pneumonia. "
        "Provide a confident description of its anatomical location, radiographic "
        "features and likely severity."
    ),
    "no_evidence": (
        "You are a senior consultant radiologist. The image shows **no evidence "
        "of pneumonia**. Provide a concise, reassuring statement in British "
        "English, highlighting clear lungs and normal mediastinal contours. "
        "Do **not** express diagnostic uncertainty or recommend further imaging."
    ),
    "unsure": (
        "You are a senior consultant radiologist. Findings are equivocal for "
        "pneumonia. Briefly state the uncertainty and outline sensible next steps."
    ),
}
USER_TEMPLATES = {
    "pneumonia": (
        "Assessment summary: This image demonstrates pneumonia.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Describe location and features."
    ),
    "no_evidence": (
        "Assessment summary: No evidence of pneumonia is seen on this image.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Reassure; do not express uncertainty."
    ),
    "unsure": (
        "Assessment summary: Findings are uncertain for pneumonia.\n\n"
        "Supporting data:\n{data}\n\n"
        "Instruction: Suggest next steps."
    ),
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
    final_label: str = Form("", description="Optional explicit label"),
    other_models: str = Form(""),
) -> JSONResponse:
    # Quick image sanity-check
    try:
        Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}") from exc

    # Collect label hints
    form = await request.form()
    raw_candidates: List[str] = []

    if final_label:
        raw_candidates.append(final_label)

    json_blob = form.get("json", "")
    try:
        blob_obj = json.loads(json_blob)
        if isinstance(blob_obj, dict) and "final_label" in blob_obj:
            raw_candidates.append(str(blob_obj["final_label"]))
        elif isinstance(blob_obj, list):
            for itm in blob_obj:
                if isinstance(itm, dict) and "final_label" in itm:
                    raw_candidates.append(str(itm["final_label"]))
    except Exception:
        pass

    raw_candidates += [other_models, form.get("other_models", "")]
    raw_candidates += labels_from_any_json(other_models or "")
    raw_candidates += labels_from_any_json(form.get("other_models", "") or "")

    chosen_label = next(
        (detect_label(c) for c in raw_candidates if detect_label(c)), "unsure"
    )
    if chosen_label == "unsure" and all(not c.strip() for c in raw_candidates):
        chosen_label = "no_evidence"

    # Optional extra context
    try:
        ctx = json.loads(other_models or json_blob or "{}")
        if not isinstance(ctx, dict):
            ctx = {}
    except Exception:
        ctx = {}

    messages = [
        {"role": "system", "content": SYS_TEMPLATES[chosen_label]},
        {
            "role": "user",
            "content": USER_TEMPLATES[chosen_label].format(
                data=json.dumps(ctx, indent=2)
            ),
        },
    ]
    answer = call_groq(messages)
    return JSONResponse({"answer": answer})


# ---------- /report ----------------------------------------------------------
@app.post("/report")
async def report_endpoint(
    file: UploadFile = File(...),
    final_label: str = Form(...),
) -> JSONResponse:
    lbl = detect_label(final_label)
    if lbl == "":
        raise HTTPException(
            400, "final_label must be pneumonia, no_evidence or unsure."
        )

    hdr_map = {
        "pneumonia": "This image demonstrates pneumonia.",
        "no_evidence": "No evidence of pneumonia is seen on this image.",
        "unsure": "Findings are uncertain for pneumonia.",
    }
    detail_map = {
        "pneumonia": "Draft 5–7 bullet points on pneumonia presence & features.",
        "no_evidence": "Draft 5–7 reassuring bullet points on normal CXR; no uncertainty.",
        "unsure": "Draft 5–7 bullet points on uncertainty & next steps.",
    }

    prompt = (
        "You are a senior consultant radiologist.\n\n"
        f"Assessment Summary: {hdr_map[lbl]}\n\n"
        f"{detail_map[lbl]}"
    )
    report = call_groq(prompt)
    return JSONResponse({"report": report, "final_label": lbl})

# ---------- /llmreport -------------------------------------------------------
class LLMReportIn(BaseModel):
    evidence: str
    summary: Optional[str] = None

@app.post("/llmreport")
async def llmreport_endpoint(payload: LLMReportIn) -> JSONResponse:
    """
    Build a consultant-style CXR report (British English) from:
      - summary: 'pneumonia' | 'no_evidence' | 'unsure'  (definitive pneumonia status)
      - evidence: text with four sections (any case):
          * THIS IS THE RESULT OF MAJORITY VOTING
          * THIS IS THE RESULT OF [CLASSIFIER]
          * THIS IS THE RESULT OF DESCRIPTIVE WITH NO DIAGNOSTIC
          * THIS IS THE RESULT OF DESCRIPTIVE WITH DIAGNOSTIC
    """
    evidence_text = (payload.evidence or "").strip()
    summary_text  = (payload.summary  or "").strip()

    if not evidence_text and not summary_text:
        raise HTTPException(400, "At least one of 'evidence' or 'summary' must be provided.")

    messages = [
        {
            "role": "system",
            "content": (
                "Act as a senior consultant radiologist. Produce ONE polished report in British English using "
                "EXACTLY these plain-text section headers and order (no markdown, no asterisks):\n"
                "Clinical details\n"
                "Technique\n"
                "Comparison\n"
                "Findings\n"
                "Impression\n"
                "Recommendations\n\n"
                "RELIABILITY HIERARCHY (highest → lowest):\n"
                "1) Pneumonia status in 'summary' (pneumonia | no_evidence | unsure) is definitive. Never contradict it.\n"
                "2) A trained medical classifier is the primary source for other thoracic labels "
                "(effusion, pneumothorax, cardiomegaly, consolidation, etc.). If it conflicts with descriptive content, prefer the classifier.\n"
                "3) The two Descriptive sections are general vision descriptions (not medical models). Use them only to refine localisation "
                "(side/zone, perihilar regions, costophrenic angles), projection/positioning, exposure, devices/lines, and clinical wording—"
                "not to introduce diagnoses that contradict 1) or 2).\n\n"
                "STYLE & CONTENT RULES:\n"
                "• Do NOT mention or allude to majority voting, models, AI, algorithms, prompts, system instructions/rules, pipelines, process language "
                "(e.g., 'asserted in accordance with the definitive summary'), chain-of-thought, raw JSON, or ANY classifier/model names "
                "(including 'CheXpert'/'Chexpert'). Write a purely clinical report.\n"
                "• Clinical details: if none are provided, write a neutral clinical question such as 'Assessment for pneumonia'.\n"
                "• Projection & consistency: if projection is stated (AP/PA/Supine), use the SAME projection throughout. "
                "If uncertain, write 'projection indeterminate' and avoid definitive heart-size calls.\n"
                "• Consistency & laterality: do NOT state a specific side AND 'side indeterminate' for the same finding—choose one. "
                "If laterality cannot be determined, write 'side indeterminate on this projection' in Findings AND repeat it in Impression.\n"
                "• Heart-size phrasing (choose ONE only): "
                "  • 'Cardiovascular/cardiomediastinal silhouette within expected limits for [projection].'  OR  "
                "  • 'Cardiac size cannot be reliably assessed on this view/projection.'  "
                "Never include both in the same report. On PA films you may comment on heart size; on AP/supine avoid definitive calls unless unequivocal. "
                "If mentioned in the Impression, keep it to a single concise clause only.\n"
                "• Technique: infer projection/positioning and exposure when possible. Acknowledge single AP/supine limitations where relevant; "
                "note that small effusions can be occult on a supine/AP film.\n"
                "• Findings – terminology hygiene: prefer 'air-space opacification' and 'interstitial change'; avoid 'infiltrate' and 'shadow'. "
                "Use 'consistent with' rather than 'diagnostic of' unless unequivocal.\n"
                "• Findings – precision: use perihilar/peribronchial terms where appropriate. Use 'consolidation' only if the classifier supports Consolidation "
                "OR descriptive evidence indicates focal dense lobar air-space opacity; otherwise keep to opacification language. Avoid vague phrases.\n"
                "• Pneumothorax & effusion — single-view limitation: when either is described on a single frontal film, append: "
                "'On this single frontal view, quantification is limited; a targeted additional view may assist.' "
                "For pneumothorax, suggest expiratory and/or lateral decubitus with the suspected side up.\n"
                "• Devices/lines – governance: if ETT/CVC/NGT are present, state presence AND measured tip position relative to landmarks "
                "(e.g., ETT tip ~2–6 cm above carina; CVC tip in SVC; NGT sub-diaphragmatic) and whether acceptable. "
                "If landmarks are not visualised, state 'landmarks not visualised—exact tip position cannot be measured on this view' (do NOT invent numbers). "
                "If malposition is suspected, flag clearly and limit recommendations to imaging/position confirmation (e.g., repeat/targeted view); "
                "do NOT propose procedure-level actions.\n"
                "• Negative path enforcement: when summary = 'no_evidence', write a confident negative Impression with no hedging, and include the phrase "
                "'within expected limits for the stated projection'.\n"
                "• Differentials: provide a differential diagnosis ONLY when summary = 'unsure', at most ONE concise differential; "
                "if summary ≠ 'unsure', do not speculate—keep the observation only.\n"
                "• Negative/normal reports: use clear, unhedged negatives (e.g., 'No pleural effusion or pneumothorax detected'); "
                "note limitations only when material to safety.\n"
                "• Recommendations (imaging-first): urgent clinical review as appropriate and bedside thoracic ultrasound to characterise pleural fluid and guide drainage if indicated. "
                "Recommend a targeted additional view when single-view limitations apply (e.g., expiratory and/or lateral decubitus). "
                "CT must NOT be suggested by default; write exactly: 'CT only if atypical/complicated or non-resolving on clinical/imaging follow-up.' "
                "Do NOT recommend medications or specific procedures (e.g., antibiotics, chest tube insertion, thoracostomy).\n"
                "• Impression: ensure the conclusion aligns with Findings without contradiction and delivers one decisive take-home message. "
                "Keep heart-size out of the Impression unless abnormal and clinically material; if included, use one concise clause only. "
                "Always mirror any key indeterminacy from Findings (e.g., laterality).\n"
                "• Do NOT include probabilities, model names, algorithm names, or raw JSON. Numerical values are permitted ONLY for device/tube position measurements. "
                "Write ~130–210 words. If inputs are incomplete, proceed with what is available.\n\n"
                "SILENT SELF-CHECK (do not print in the report): before finalising, verify that (a) there is no 'side' vs 'indeterminate' contradiction; "
                "(b) heart-size uses ONE phrasing only; (c) differentials appear only when summary = 'unsure'; "
                "(d) device tips are measured only if landmarks are visualised, otherwise state not visualised; "
                "(e) the single-view limitation clause is present when relevant; "
                "(f) the CT line reads exactly: 'CT only if atypical/complicated or non-resolving on clinical/imaging follow-up.'"
            ),
        },
        {
            "role": "user",
            "content": (
                "### Majority decision summary (one of: pneumonia | no_evidence | unsure)\n"
                "```text\n"
                f"{summary_text or 'unsure'}\n"
                "```\n\n"
                "### Evidence (four sections; headings may be any case)\n"
                "```text\n"
                f"{evidence_text or 'None provided'}\n"
                "```"
            ),
        },
    ]

    report = call_groq(
        messages,
        model=os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b"),
        temperature=float(os.getenv("GROQ_TEMP", "0.35")),
        top_p=float(os.getenv("GROQ_TOP_P", "1")),
        max_completion_tokens=int(os.getenv("GROQ_MAX_REPORT_TOKENS", "8192")),
        stream=True,  # important for gpt-oss-120b
        reasoning_effort=os.getenv("GROQ_REASONING", "medium"),
    )
    return JSONResponse({"report": report})


# ---------- /analyse (Maverick, descriptive only, no diagnosis) --------------
@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    extra_instructions: str = Form(
        "",
        description="Optional extra guidance (kept non-diagnostic).",
    ),
) -> JSONResponse:
    """
    Non-diagnostic descriptive analysis of a chest X-ray using Groq Maverick.
    Returns structured JSON. Mentions orientation markers and precise lung
    locations where applicable. No diagnosis or clinical advice.
    """
    try:
        img_bytes: bytes = await file.read()
        Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(400, f"Bad image: {exc}") from exc
    except Exception as exc:
        raise HTTPException(400, f"Unable to read image: {exc!r}") from exc

    data_url = make_data_url_under_limit(img_bytes, file.filename or "upload.png")

    json_request_instructions = f"""
You are an expert radiography describer. Produce a purely descriptive account
in British English of the chest X-ray shown. Do not provide a diagnosis,
probability, recommendation, or clinical certainty language. Focus strictly on
what is visible. When relevant, name precise lung locations (e.g., right upper
zone, left lower zone, perihilar region, costophrenic angle).
Return a strict JSON object with exactly these keys:
{{
  "image_orientation_marker": "One of: 'R', 'L', 'Both', 'None visible', or 'Unclear'.",
  "view_and_positioning": "Short text (e.g., PA/AP/Supine/Erect if inferable; otherwise 'Unclear').",
  "exposure_contrast": "Short text on exposure/contrast (e.g., under-/over-exposed, adequate), citing where applicable.",
  "anatomical_description": "Short paragraph describing visible anatomy and lung fields, mentioning specific lung locations when appropriate.",
  "devices_and_artefacts": ["List devices/artefacts/implants/foreign bodies if seen; otherwise empty array."],
  "suspected_structural_changes": "Short text noting any visual discontinuities/irregularities/fracture-like lines or asymmetries WITHOUT naming a diagnosis; include locations if relevant.",
  "overall_description": "2–3 sentences giving a plain-English, non-diagnostic overview of what the image depicts."
}}
{extra_instructions.strip()}
""".strip()

    # IMPORTANT: Groq expects 'text' and 'image_url' with nested {'url': ...}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json_request_instructions},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    raw = call_groq(
        messages,
        model=GROQ_VISION_MODEL_MAVERICK,
        temperature=0.2,
        top_p=1,
        response_format={"type": "json_object"},
        max_completion_tokens=1024,
    )

    # Try to parse JSON strictly; if it fails, return raw text so the client can see why
    parsed: Optional[Dict[str, Any]] = None
    try:
        maybe = json.loads(raw)
        if isinstance(maybe, dict):
            parsed = maybe
    except Exception:
        parsed = None

    return JSONResponse(
        {
            "model": GROQ_VISION_MODEL_MAVERICK,
            "analysis": parsed,
            "raw": None if parsed is not None else raw,
            "note": "Descriptive only. No diagnostic interpretation provided.",
        }
    )
    
# ---------- /diagnose (descriptive + diagnostic) -----------------------------
@app.post("/diagnose")
async def diagnose(
    file: UploadFile = File(...),
    extra_instructions: str = Form(
        "",
        description="Optional extra guidance (diagnostic allowed).",
    ),
) -> JSONResponse:
    """
    Diagnostic analysis of a chest X-ray using Groq Maverick.
    Returns structured JSON with descriptive fields AND per-label diagnostic calls.
    This endpoint is for research/informational purposes and not for clinical use.
    """
    try:
        img_bytes: bytes = await file.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(400, f"Bad image: {exc}") from exc
    except Exception as exc:
        raise HTTPException(400, f"Unable to read image: {exc!r}") from exc

    # Create compact data URL for the vision model
    data_url = make_data_url_under_limit(img_bytes, file.filename or "upload.png")

    # Optional: include CNN evidence as structured context (model never sees the raw image pixels here)
    try:
        cnn = classify_image(pil)
    except Exception:
        cnn = {"note": "CNN evidence unavailable for this image."}

    # Build the diagnostic JSON instruction
    # NB: Ask for explicit status + confidence per label and short evidence with location terms.
    label_list_json = json.dumps(LABELS, ensure_ascii=False)
    json_request_instructions = f"""
You are a senior consultant radiologist. Describe and DIAGNOSE the chest X-ray in British English.
Use precise anatomical terms (e.g., right upper zone, left lower zone, perihilar, costophrenic angle).
Consider the image FIRST; you may use the supporting CNN evidence as secondary context.

Return a strict JSON object with exactly these keys and shapes:
{{
  "image_orientation_marker": "One of: 'R', 'L', 'Both', 'None visible', or 'Unclear'.",
  "view_and_positioning": "PA/AP/Supine/Erect if inferable; otherwise 'Unclear'.",
  "exposure_contrast": "Short text on exposure/contrast adequacy.",
  "anatomical_description": "Short paragraph describing visible anatomy and lung fields.",
  "devices_and_artefacts": ["List devices/artefacts/implants/foreign bodies if seen; otherwise empty array."],
  "findings": {{
    "labels": {label_list_json},
    "per_label": {{
      "<label>": {{
        "status": "present" | "absent" | "uncertain",
        "confidence": 0.00-1.00,
        "location": "Anatomical location(s) or 'None'",
        "evidence": "One or two sentences citing visible signs (e.g., focal opacity at LLL, blunted R CPA).",
        "severity": "If applicable: mild | moderate | severe | 'N/A'"
      }}
      // Include an entry for EVERY label in 'labels' above.
    }}
  }},
  "overall_impression": "≤ 5 sentences summarising the key diagnoses and their clinical significance.",
  "recommendations": ["0–3 concise next steps if relevant; otherwise empty array."],
  "disclaimer": "For research/informational use only; not a clinical decision tool."
}}
Special handling for 'No Finding': set status 'present' ONLY when all other labels are 'absent' and exposure/positioning are adequate; otherwise set 'absent'.
{extra_instructions.strip()}
""".strip()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        json_request_instructions
                        + "\n\n### Supporting CNN evidence (optional)\n```json\n"
                        + json.dumps(cnn, indent=2)
                        + "\n```"
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    raw = call_groq(
        messages,
        model=GROQ_VISION_MODEL_MAVERICK,
        temperature=0.2,
        top_p=1,
        response_format={"type": "json_object"},
        max_completion_tokens=1536,
    )

    parsed: Optional[Dict[str, Any]] = None
    try:
        maybe = json.loads(raw)
        if isinstance(maybe, dict):
            parsed = maybe
    except Exception:
        parsed = None

    return JSONResponse(
        {
            "model": GROQ_VISION_MODEL_MAVERICK,
            "diagnosis": parsed,
            "raw": None if parsed is not None else raw,
            "note": "Diagnostic output for research/informational use only.",
        }
    )
