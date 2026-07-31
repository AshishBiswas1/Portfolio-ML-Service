import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "inquiry_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

CLASSES = np.array(['Hiring/Recruiter', 'Project Work', 'General Question', 'Spam'])

class OnlineInquiryTrainer:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.vectorizer = None
        self.classifier = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            try:
                self.classifier = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                return
            except Exception as e:
                print(f"[OnlineTrainer Warning]: Failed to load existing model: {e}")

        # Baseline Initialization
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.classifier = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=1000, random_state=42)
        self._train_baseline()

    def _train_baseline(self):
        training_corpus = [
            # Hiring / Recruiter
            ("we are hiring for a senior full stack engineer position at our company", "Hiring/Recruiter"),
            ("interview invitation for backend developer job role", "Hiring/Recruiter"),
            ("technical recruiter reaching out regarding open position salary", "Hiring/Recruiter"),
            ("looking for AI ML specialist to join our engineering team fulltime", "Hiring/Recruiter"),
            ("career opportunity role remote position hiring manager", "Hiring/Recruiter"),
            ("job offer software engineer compensation package details", "Hiring/Recruiter"),

            # Project Work / Freelance
            ("freelance client quote for web application build and deployment", "Project Work"),
            ("contract project development quote estimate budget timeline", "Project Work"),
            ("need a developer to design and deploy full stack SaaS app", "Project Work"),
            ("client inquiry regarding software consultancy and build project", "Project Work"),
            ("freelance work project timeline hourly rate per project", "Project Work"),

            # General Question
            ("question about your portfolio tech stack architecture and design", "General Question"),
            ("admiring your projects and wanted to say hello developer", "General Question"),
            ("inquiry regarding your open source repository on github", "General Question"),
            ("quick feedback on your Next.js and Node.js implementation", "General Question"),

            # Spam
            ("buy cheap backlinks boost website SEO rank fast instant", "Spam"),
            ("crypto token pre-sale investment guaranteed ROI 100x bonus", "Spam"),
            ("click this link to claim free prize money reward cash", "Spam"),
            ("casino slots betting leverage win instant cash bonus", "Spam")
        ]

        texts, labels = zip(*training_corpus)
        X_vec = self.vectorizer.fit_transform(texts)
        self.classifier.partial_fit(X_vec, labels, classes=CLASSES)
        self._save_checkpoint()

    def _save_checkpoint(self):
        joblib.dump(self.classifier, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)

    def predict(self, subject: str, message: str) -> dict:
        full_text = f"{subject} {message}".strip().lower()
        if not full_text:
            return {
                "intent": "General Question",
                "confidenceScore": 0.85,
                "priority": "Medium",
                "sentimentScore": 0.5,
                "keywords": []
            }

        X_vec = self.vectorizer.transform([full_text])
        probs = self.classifier.predict_proba(X_vec)[0]
        max_idx = np.argmax(probs)
        intent = str(CLASSES[max_idx])
        confidence = float(probs[max_idx])

        # Rule-based priority & sentiment override heuristics
        hiring_kws = ['job', 'hire', 'interview', 'role', 'salary', 'recruiter', 'position', 'full-time', 'offer']
        project_kws = ['freelance', 'build', 'project', 'client', 'quote', 'contract', 'develop']
        spam_kws = ['crypto', 'seo', 'backlinks', 'prize', 'bonus', 'casino']

        matched_hiring = [kw for kw in hiring_kws if kw in full_text]
        matched_project = [kw for kw in project_kws if kw in full_text]
        matched_spam = [kw for kw in spam_kws if kw in full_text]

        if matched_spam and intent != 'Spam':
            intent = 'Spam'
        elif matched_hiring and intent != 'Hiring/Recruiter':
            intent = 'Hiring/Recruiter'
        elif matched_project and intent != 'Project Work':
            intent = 'Project Work'

        priority = "High" if intent in ['Hiring/Recruiter', 'Project Work'] else "Low" if intent == 'Spam' else "Medium"
        sentiment = 0.9 if intent == 'Hiring/Recruiter' else 0.8 if intent == 'Project Work' else -0.5 if intent == 'Spam' else 0.5
        keywords = list(set(matched_hiring + matched_project + matched_spam))

        return {
            "intent": intent,
            "confidenceScore": round(confidence, 2),
            "priority": priority,
            "sentimentScore": sentiment,
            "keywords": keywords
        }

    def partial_fit_update(self, text: str, corrected_intent: str) -> dict:
        """
        Online Incremental Learning Endpoint: Updates Model 2 weights in real time based on feedback
        """
        if corrected_intent not in CLASSES:
            return {"status": "error", "message": f"Invalid intent label. Must be one of {list(CLASSES)}"}

        clean_text = text.strip().lower()
        if not clean_text:
            return {"status": "error", "message": "Text payload cannot be empty"}

        X_vec = self.vectorizer.transform([clean_text])
        self.classifier.partial_fit(X_vec, [corrected_intent])
        self._save_checkpoint()

        return {
            "status": "success",
            "message": f"Model 2 weights updated incrementally for intent '{corrected_intent}'",
            "updated_checkpoint": MODEL_PATH
        }

online_trainer = OnlineInquiryTrainer()
