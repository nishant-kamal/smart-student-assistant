import os, time, json, io
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

app = FastAPI(title="Smart Student Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Logging path (backend/logs folder) — resolved relative to this file,
# not the process's current working directory, so it works the same
# whether run via `uvicorn backend.app:app` from /app or locally from backend/.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "requests.log")

def log_event(entry):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # ignore logging errors

# =============================
# Load Fine-Tuned Model If Exists
# =============================
FINETUNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert-qa-final")

print("Startup: checking models...")

qa_model = None
if os.path.exists(FINETUNED_MODEL_PATH):
    try:
        print(f" Loading fine-tuned model from {FINETUNED_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        model = AutoModelForQuestionAnswering.from_pretrained(FINETUNED_MODEL_PATH)
        qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)
    except Exception as e:
        # Fine-tuned artifact exists but failed to load (e.g. corrupted or
        # incomplete weights file). Don't take the whole API down for this —
        # fall back to the public default model instead.
        print(f"⚠️ Failed to load fine-tuned model ({e}); falling back to default model.")
        qa_model = None

if qa_model is None:
    print("ℹ️ Fine-tuned model not found or failed to load — using default model.")
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Summarizer Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Data model
class QARequest(BaseModel):
    context: str
    question: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/qa")
async def qa(req: QARequest, request: Request):
    start = time.time()
    try:
        result = qa_model(question=req.question, context=req.context)
        answer = result["answer"].strip()
        success = True
    except Exception as e:
        answer = f"Error: {str(e)}"
        success = False

    log_event({
        "endpoint": "/qa",
        "question": req.question,
        "context_length": len(req.context),
        "answer": answer,
        "latency": round(time.time() - start, 4),
        "success": success
    })
    return {"answer": answer}

@app.post("/summarize")
async def summarize(data: dict):
    text = data.get("text") or data.get("context") or ""
    start = time.time()
    try:
        summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
        success = True
    except Exception as e:
        summary = f"Error: {str(e)}"
        success = False

    log_event({
        "endpoint": "/summarize",
        "input_length": len(text),
        "summary": summary,
        "latency": round(time.time() - start, 4),
        "success": success
    })
    return {"summary": summary}

@app.post("/hw-eval")
async def handwriting_eval(file: UploadFile = File(...)):
    start = time.time()
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        extracted_text = pytesseract.image_to_string(image)
        success = True
    except Exception as e:
        extracted_text = f"Error processing image: {e}"
        success = False

    log_event({
        "endpoint": "/hw-eval",
        "file_size": len(image_bytes),
        "text_length": len(extracted_text),
        "latency": round(time.time() - start, 4),
        "success": success
    })
    return {"extracted_text": extracted_text}
