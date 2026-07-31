# Portfolio Machine Learning Microservice (`Portfolio-ML-Service`)

A high-performance Python FastAPI Machine Learning microservice providing:
1. **Model 1 (Sentence Transformer Embeddings):** Dense 384-dimensional text embeddings (`all-MiniLM-L6-v2`) for semantic search and recommendation matching.
2. **Model 2 (Continuous Online Learning Classifier):** Real-time incremental `partial_fit()` SGDClassifier for contact message intent & priority tagging.
3. **Model 2 (Visitor Telemetry & Inter-Model Feedback Loop):** Dynamic visitor persona tracking (scroll depth, dwell time, section hovers) that dynamically biases Model 1 recommendations in real-time.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 2. Run Automated Unit & Integration Tests
```bash
python test_models.py
```

### 3. Start FastAPI ML Server
```bash
python main.py
# Server runs at http://localhost:5000
```

---

## 📡 API Endpoints

- `GET /health` — Microservice health status.
- `POST /ml/embeddings` — Returns 384-dimensional vector embedding for text.
- `POST /ml/predict/intent` — Evaluates message intent (*Hiring/Recruiter*, *Project Work*, *General Question*, *Spam*) & priority.
- `POST /ml/feedback/intent` — Online `partial_fit()` endpoint to dynamically retrain Model 2 live from admin feedback.
- `POST /ml/telemetry/track` — Process visitor interaction telemetry and update session persona.
- `POST /ml/personalized-recommendations` — Inter-Model Feedback endpoint blending Model 2 persona vector with Model 1 vector similarity.