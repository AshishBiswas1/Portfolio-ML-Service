import numpy as np

class TelemetryPersonaEngine:
    def __init__(self):
        # Active session store: { session_id: { persona_vector: np.array, events: [] } }
        self.sessions = {}

    def track_event(self, session_id: str, event_type: str, payload: dict) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "persona_vector": np.zeros(384, dtype=np.float32),
                "interests": {"aiml": 0.0, "backend": 0.0, "frontend": 0.0, "fullstack": 0.0},
                "event_count": 0
            }

        session = self.sessions[session_id]
        session["event_count"] += 1

        # Evaluate interaction weights
        path = str(payload.get("path", "")).lower()
        techs = [t.lower() for t in payload.get("technologies", [])]
        dwell_time = float(payload.get("dwell_time_seconds", 1.0))

        if "ai" in path or "ml" in path or any(t in ["pytorch", "tensorflow", "onnx", "nlp", "vector", "python"] for t in techs):
            session["interests"]["aiml"] += 1.5 * (1.0 + dwell_time / 10.0)
        if "backend" in path or any(t in ["node", "express", "mongodb", "docker", "sql", "api"] for t in techs):
            session["interests"]["backend"] += 1.2 * (1.0 + dwell_time / 10.0)
        if "frontend" in path or any(t in ["react", "next.js", "tailwind", "three.js", "framer"] for t in techs):
            session["interests"]["frontend"] += 1.0 * (1.0 + dwell_time / 10.0)

        session["interests"]["fullstack"] = (session["interests"]["backend"] + session["interests"]["frontend"]) / 2.0

        # Determine dominant inferred persona
        dominant_role = max(session["interests"], key=session["interests"].get)

        return {
            "session_id": session_id,
            "inferred_role": dominant_role,
            "interest_scores": session["interests"],
            "total_events": session["event_count"],
            "autonomous_db_persistence": {
                "triggered": True,
                "target_persona": dominant_role,
                "action": "AUTOMATIC_DATABASE_VALUE_MUTATION"
            }
        }

    def get_inferred_role(self, session_id: str = None) -> str:
        if session_id and session_id in self.sessions:
            scores = self.sessions[session_id]["interests"]
            dominant = max(scores, key=scores.get)
            if scores[dominant] > 0.5:
                return dominant
        return "fullstack"

    def compute_personalized_rankings(self, target_embedding: list, candidate_projects: list, session_id: str = None, alpha: float = 0.7, beta: float = 0.3) -> list:
        """
        Inter-Model Feedback Loop (Model 2 -> Model 1):
        Combines target project embedding (Model 1) with visitor persona bias (Model 2)
        to dynamically re-rank recommendations.
        """
        target_vec = np.array(target_embedding, dtype=np.float32)
        target_norm = np.linalg.norm(target_vec)
        if target_norm > 0:
            target_vec = target_vec / target_norm

        role = self.get_inferred_role(session_id)

        scored_candidates = []
        for p in candidate_projects:
            p_vec = np.array(p.get("embedding", []), dtype=np.float32)
            if len(p_vec) != 384:
                p_vec = np.zeros(384, dtype=np.float32)

            p_norm = np.linalg.norm(p_vec)
            if p_norm > 0:
                p_vec = p_vec / p_norm

            cos_sim = float(np.dot(target_vec, p_vec))
            role_score = float(p.get("roleScores", {}).get(role, 85.0)) / 100.0

            # Combined biased score
            final_score = (alpha * cos_sim) + (beta * role_score)

            p_copy = {k: v for k, v in p.items() if k != "embedding"}
            p_copy["similarity"] = round(cos_sim, 4)
            p_copy["personalized_score"] = round(final_score, 4)
            p_copy["inferred_role"] = role
            scored_candidates.append(p_copy)

        scored_candidates.sort(key=lambda x: x["personalized_score"], reverse=True)
        return scored_candidates

telemetry_engine = TelemetryPersonaEngine()
