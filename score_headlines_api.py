# score_headlines_api.py  (colócalo junto a svm.joblib)
import logging, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Carga de modelo y encoder (una sola vez)
MODEL = joblib.load("svm.joblib")
ENCODER = SentenceTransformer("all-MiniLM-L6-v2")

class HeadlinesRequest(BaseModel):
    headlines: List[str]

@app.get("/status")
def status():
    logger.info("Health-check")
    return {"status": "OK"}

@app.post("/score_headlines")
def score_headlines(req: HeadlinesRequest):
    try:
        logger.info("Scoring %d headlines", len(req.headlines))
        emb = ENCODER.encode(req.headlines)
        preds = MODEL.predict(emb).tolist()
        return {"labels": preds}
    except Exception as exc:
        logger.error("Scoring failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
