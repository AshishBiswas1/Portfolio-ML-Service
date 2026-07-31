import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_model.pt")

CLASSES = ['Hiring/Recruiter', 'Project Work', 'General Question', 'Spam']
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

class PortfolioPyTorchNet(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_classes=4):
        super(PortfolioPyTorchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class PyTorchEngine:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PortfolioPyTorchNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(PYTORCH_MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=self.device))
                self.model.eval()
                print("[PyTorchEngine]: Loaded existing PyTorch model state_dict.")
                return
            except Exception as e:
                print(f"[PyTorchEngine Warning]: Failed to load state_dict: {e}")

        # Baseline PyTorch Training Step
        self._train_baseline()

    def _train_baseline(self):
        self.model.train()
        # Synthetic baseline training tensors for PyTorch initialization
        inputs = torch.randn(32, 384, device=self.device)
        targets = torch.randint(0, 4, (32,), device=self.device)

        for _ in range(50):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

        self.save_weights()

    def save_weights(self):
        torch.save(self.model.state_dict(), PYTORCH_MODEL_PATH)

    def predict_torch(self, vector: list) -> dict:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor([vector], dtype=torch.float32, device=self.device)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        max_idx = int(np.argmax(probs))
        predicted_class = CLASSES[max_idx]
        confidence = float(probs[max_idx])

        return {
            "pytorch_intent": predicted_class,
            "pytorch_confidence": round(confidence, 4),
            "class_probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}
        }

    def train_torch_step(self, vector: list, label: str) -> dict:
        if label not in CLASS_MAP:
            return {"status": "error", "message": f"Invalid label. Must be one of {CLASSES}"}

        self.model.train()
        x_tensor = torch.tensor([vector], dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor([CLASS_MAP[label]], dtype=torch.long, device=self.device)

        self.optimizer.zero_grad()
        output = self.model(x_tensor)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()

        self.save_weights()

        return {
            "status": "success",
            "message": "PyTorch model parameters updated via backpropagation step",
            "loss": float(loss.item())
        }

pytorch_engine = PyTorchEngine()
