# Ashish Biswas Portfolio - ML Microservice (`Portfolio-ML-Service`)

## Project Overview
The **`Portfolio-ML-Service`** repository is a specialized Python machine learning microservice built with **FastAPI**, **Uvicorn**, **PyTorch**, **Sentence-Transformers**, and **Scikit-learn**. It provides intelligence capabilities for the portfolio platform, including semantic text embeddings, project relevance scoring, contact message intent classification, and real-time visitor persona tracking.

---

## How It Works
1. **Sentence Transformer Embeddings (Model 1)**: Generates 384-dimensional dense vector embeddings using `all-MiniLM-L6-v2` for semantic text similarity and project relevance matching.
2. **Continuous Online Learning Classifier (Model 2)**: Utilizes an incremental `partial_fit()` SGDClassifier to classify contact message intent (*Hiring/Recruiter*, *Project Work*, *General Question*, *Spam*) and assign priority ratings.
3. **Visitor Telemetry & Persona Engine**: Processes visitor interaction signals (dwell time, scroll depth, section hovers) to dynamically calculate session interest vectors.
4. **Inter-Model Feedback Loop**: Blends Model 2 visitor persona vectors with Model 1 project embeddings to generate dynamic personalized recommendations.

---

## Built Features
- **Semantic Text Embeddings Endpoint (`POST /ml/embeddings`)**: Computes dense vector representations for project descriptions and technical summaries.
- **Intent & Priority Prediction (`POST /ml/predict/intent`)**: Classifies incoming contact messages into structured intent categories with confidence scores.
- **Live Online Learning Feedback (`POST /ml/feedback/intent`)**: Retrains the intent classifier in real time from administrator feedback without server restarts.
- **Visitor Telemetry Processor (`POST /ml/telemetry/track`)**: Analyzes visitor navigation behavior to derive real-time session personas.
- **Personalized Recommendations Engine (`POST /ml/personalized-recommendations`)**: Computes cosine similarity between visitor intent vectors and project feature vectors.
- **Automated Test Suite (`test_models.py`)**: Unit and integration tests covering vector dimensions, inference speeds, and feedback loops.

---

## Website Description
This microservice acts as the analytical brain of the website backend. It transforms static portfolio content into an intelligent showcase by evaluating the technical complexity of projects, scoring skill relevance, prioritizing recruiter messages, and personalizing project recommendations based on visitor behavior.

---

## Environment Variables & Security Note

> [!NOTE]  
> All sensitive configuration keys, internal credentials, and environment settings have been omitted from this documentation and source code to ensure security compliance.

### Required Environment Variables (Structure Only)
```env
PORT=<MICROSERVICE_PORT>
PYTHONUNBUFFERED=1
```