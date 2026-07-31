import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainers.online_trainer import online_trainer
from trainers.telemetry_engine import telemetry_engine
from trainers.pytorch_engine import pytorch_engine
from trainers.resume_skills_engine import resume_skills_engine

def run_tests():
    print("=========================================================")
    print("TESTING PORTFOLIO PYTORCH & CONTINUOUS ONLINE ML MODELS")
    print("=========================================================")

    # 1. Test Scikit-Learn Model 2 Baseline Intent Prediction
    sample_subject = "Interview Invitation"
    sample_msg = "We are impressed with your portfolio and would like to offer a full-time senior role"
    pred1 = online_trainer.predict(sample_subject, sample_msg)
    print(f"\n[Test 1] Baseline Scikit-Learn Intent Prediction:")
    print(f"Result: Intent={pred1['intent']}, Priority={pred1['priority']}, Confidence={pred1['confidenceScore']}")
    assert pred1['intent'] == 'Hiring/Recruiter'

    # 2. Test PyTorch Neural Network Forward Pass Inference
    print(f"\n[Test 2] Testing PyTorch Neural Network Forward Pass (PortfolioPyTorchNet)...")
    fake_vector = [0.05] * 384
    pytorch_pred = pytorch_engine.predict_torch(fake_vector)
    print(f"PyTorch Intent Prediction: {pytorch_pred['pytorch_intent']} (Conf: {pytorch_pred['pytorch_confidence']})")
    assert "pytorch_intent" in pytorch_pred

    # 3. Test PyTorch Gradient Optimization & Backpropagation Step
    print(f"\n[Test 3] Testing PyTorch Backpropagation & Gradient Update Step...")
    torch_step = pytorch_engine.train_torch_step(fake_vector, "Hiring/Recruiter")
    print(f"PyTorch Backprop Step: Status={torch_step['status']}, Loss={torch_step['loss']:.4f}")
    assert torch_step['status'] == 'success'

    # 4. Test Scikit-Learn Online Incremental Learning (partial_fit)
    print(f"\n[Test 4] Testing Online Incremental Learning (partial_fit)...")
    feedback_text = "We have an urgent contract build project for a web app client"
    feedback_res = online_trainer.partial_fit_update(feedback_text, "Project Work")
    print(f"Feedback Status: {feedback_res['status']} | Message: {feedback_res['message']}")
    pred2 = online_trainer.predict("New Project Inquiry", feedback_text)
    assert pred2['intent'] == 'Project Work'

    # 5. Test Visitor Telemetry Tracking & Persona Vector (Model 2)
    print(f"\n[Test 5] Testing Visitor Telemetry Tracking & Persona Bias...")
    session_id = "test_visitor_session_001"
    persona = telemetry_engine.track_event(session_id, "page_view", {
        "path": "/projects/ai-rag-pipeline",
        "technologies": ["PyTorch", "NLP", "Vector DB", "Python"],
        "dwell_time_seconds": 15.0
    })
    print(f"Calculated Visitor Persona: {persona['inferred_role']}")
    assert persona['inferred_role'] == 'aiml'

    # 6. Test Targeted Resume Summary Adaptation
    print(f"\n[Test 6] Testing Targeted Resume Summary Adaptation...")
    summary_ai = resume_skills_engine.get_targeted_summary("aiml")
    print(f"Targeted AI/ML Summary: {summary_ai['targetedSummary'][:80]}...")
    assert "AI/ML" in summary_ai['targetedSummary']

    # 7. Test Skill & Internship Role Affinity Ranking
    print(f"\n[Test 7] Testing Skill & Internship Role Affinity Ranking...")
    skills_sample = [{"name": "PyTorch"}, {"name": "React.js"}, {"name": "Node.js"}]
    ranked_skills = resume_skills_engine.rank_skills(skills_sample, "aiml")
    print(f"Top Ranked Skill for AI/ML: {ranked_skills[0]['name']}")
    assert ranked_skills[0]['name'] == 'PyTorch'

    print("\n=========================================================")
    print("ALL ML MODELS, TARGETED RESUME & SKILL ENGINES PASSED!")
    print("=========================================================")

if __name__ == "__main__":
    run_tests()
