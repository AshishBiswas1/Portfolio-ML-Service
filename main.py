import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from trainers.online_trainer import online_trainer
from trainers.telemetry_engine import telemetry_engine
from trainers.pytorch_engine import pytorch_engine
from trainers.resume_skills_engine import resume_skills_engine

app = FastAPI(
    title="Portfolio Multi-Model PyTorch & ML Service",
    description="Python FastAPI Microservice featuring PyTorch Deep Neural Networks, Scikit-Learn Online Learning & Autonomous Visitor Personalization",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODEL 1: LIGHTWEIGHT 384-DIM DENSE EMBEDDING ENGINE ───
class LightweightEmbeddingEngine:
    def __init__(self, n_components=384):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fit_corpus()

    def _fit_corpus(self):
        corpus = [
            "React Next.js Tailwind CSS Frontend Web Application User Interface",
            "Node.js Express MongoDB REST API Microservice Backend System Database",
            "Python Machine Learning PyTorch Scikit-Learn Artificial Intelligence Vector Embeddings",
            "Full Stack Software Engineer System Architecture Cloud Deployment Docker",
            "Database Schema Design Mongoose Authentication JWT Security",
            "Sentence Transformers ONNX Natural Language Processing Intent Classification",
            "Portfolio Interactive Web App Visual Design Performance Optimization"
        ]
        X = self.vectorizer.fit_transform(corpus)
        if X.shape[0] < self.n_components:
            self.basis = np.random.RandomState(42).randn(2000, self.n_components).astype(np.float32)

    def encode(self, text: str) -> List[float]:
        clean_text = text.strip().lower()
        if not clean_text:
            return [0.0] * self.n_components

        X_vec = self.vectorizer.transform([clean_text])
        if hasattr(self, 'basis') and X_vec.shape[1] == self.basis.shape[0]:
            dense = X_vec.toarray() @ self.basis
            vec = dense[0]
        else:
            hash_seed = abs(hash(clean_text)) % 1000000
            vec = np.random.RandomState(hash_seed).randn(self.n_components).astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

embedding_engine = LightweightEmbeddingEngine(n_components=384)

# ─── PYDANTIC SCHEMAS ───
class EmbeddingRequest(BaseModel):
    text: str

class IntentPredictRequest(BaseModel):
    subject: Optional[str] = ""
    message: str

class IntentFeedbackRequest(BaseModel):
    text: str
    corrected_intent: str

class TelemetryTrackRequest(BaseModel):
    session_id: str
    path: Optional[str] = "/"
    technologies: Optional[List[str]] = []
    dwell_time_seconds: Optional[float] = 1.0

class PersonalizedRecRequest(BaseModel):
    target_project_id: str
    target_embedding: List[float]
    candidate_projects: List[dict]
    session_id: Optional[str] = None

class TargetedSummaryRequest(BaseModel):
    role: Optional[str] = None
    session_id: Optional[str] = None

class RankItemsRequest(BaseModel):
    items: List[dict]
    role: Optional[str] = None
    session_id: Optional[str] = None

# ─── API ENDPOINTS ───

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Portfolio-ML-Service",
        "pytorch_version": torch.__version__,
        "model1": "384-dim Dense Vector Embedding Engine",
        "model2": "PyTorch Neural Network (PortfolioPyTorchNet) + Scikit-Learn SGDClassifier",
        "personalization": "Autonomous Visitor Telemetry & Dynamic Role Engine"
    }

@app.post("/ml/embeddings")
def generate_embedding(req: EmbeddingRequest):
    vector = embedding_engine.encode(req.text)
    return {"embedding": vector, "dimensions": len(vector)}

@app.post("/ml/predict/intent")
def predict_inquiry_intent(req: IntentPredictRequest):
    scikit_res = online_trainer.predict(req.subject or "", req.message)
    vector = embedding_engine.encode(f"{req.subject} {req.message}")
    pytorch_res = pytorch_engine.predict_torch(vector)

    return {
        "status": "success",
        "analysis": scikit_res,
        "pytorch_analysis": pytorch_res
    }

@app.post("/ml/feedback/intent")
def update_inquiry_feedback(req: IntentFeedbackRequest):
    scikit_res = online_trainer.partial_fit_update(req.text, req.corrected_intent)
    vector = embedding_engine.encode(req.text)
    pytorch_res = pytorch_engine.train_torch_step(vector, req.corrected_intent)

    if scikit_res.get("status") == "error":
        raise HTTPException(status_code=400, detail=scikit_res.get("message"))

    return {
        "status": "success",
        "scikit_result": scikit_res,
        "pytorch_result": pytorch_res
    }

@app.post("/ml/telemetry/track")
def track_visitor_telemetry(req: TelemetryTrackRequest):
    res = telemetry_engine.track_event(
        session_id=req.session_id,
        event_type="page_view",
        payload={
            "path": req.path,
            "technologies": req.technologies,
            "dwell_time_seconds": req.dwell_time_seconds
        }
    )
    return {"status": "success", "persona": res}

@app.post("/ml/personalized-recommendations")
def get_personalized_recommendations(req: PersonalizedRecRequest):
    ranked = telemetry_engine.compute_personalized_rankings(
        target_embedding=req.target_embedding,
        candidate_projects=req.candidate_projects,
        session_id=req.session_id
    )
    return {"status": "success", "recommendations": ranked[:3]}

@app.post("/ml/targeted-summary")
def get_targeted_summary(req: TargetedSummaryRequest):
    role = req.role or telemetry_engine.get_inferred_role(req.session_id)
    return {"status": "success", "inferred_role": role, "data": resume_skills_engine.get_targeted_summary(role)}

@app.post("/ml/rank/skills")
def rank_skills_by_role(req: RankItemsRequest):
    role = req.role or telemetry_engine.get_inferred_role(req.session_id)
    ranked = resume_skills_engine.rank_skills(req.items, role)
    return {"status": "success", "inferred_role": role, "skills": ranked}

@app.post("/ml/rank/internships")
def rank_internships_by_role(req: RankItemsRequest):
    role = req.role or telemetry_engine.get_inferred_role(req.session_id)
    ranked = resume_skills_engine.rank_internships(req.items, role)
    return {"status": "success", "inferred_role": role, "internships": ranked}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
