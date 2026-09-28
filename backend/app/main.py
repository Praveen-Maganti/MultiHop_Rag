from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.experiments import router as experiments_router
from app.api.metrics import router as metrics_router

app = FastAPI(title="Adaptive Memory RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api/documents", tags=["documents"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(experiments_router, prefix="/api/experiments", tags=["experiments"])
app.include_router(metrics_router, prefix="/api/metrics", tags=["metrics"])

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Adaptive Memory RAG API is running"}
