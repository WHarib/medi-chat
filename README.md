title: Medi-Chat API (CheXpert + Groq)
emoji: 🩻
colorFrom: teal
colorTo: gray
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---

# Medi-Chat API (CheXpert + Groq)

FastAPI microservice that:
1. Predicts **14 CheXpert chest X-ray labels** using a fine-tuned DenseNet121 (`densenet121_finetuned.pth`).
2. Optionally calls **Groq (llama3-70b-8192)** to answer a free-text medical question based on those predictions.

## Endpoints

| Method | Path               | Body                         | Returns |
|--------|--------------------|------------------------------|---------|
| POST   | `/predict_chexpert`| `multipart/form-data`, field `file` | JSON with probs/thresholds/detected labels |
| POST   | `/chat`            | `multipart/form-data`: `file`, and form field `question` | JSON with LLM `answer` + predictions |

### Example (curl)

```bash
curl -F "file=@xray.png" https://<user>-medi-chat.hf.space/predict_chexpert
curl -F "file=@xray.png" -F "question=What does this mean?" https://<user>-medi-chat.hf.space/chat