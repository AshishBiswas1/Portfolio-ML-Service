class ResumeSkillsPersonalizer:
    def __init__(self):
        self.role_summaries = {
            "aiml": "Results-driven AI/ML Engineer specializing in Sentence Transformers, PyTorch deep learning, vector search embeddings, and NLP pipelines. Experienced in designing continuous online-learning microservices and intelligent recommendation engines.",
            "backend": "High-throughput Backend Architect proficient in Node.js, Express, MongoDB Atlas, REST APIs, and microservices. Experienced in designing robust API gateways, database vector indexing, and scalable server architecture.",
            "frontend": "Creative Full-Stack & Frontend Engineer passionate about Next.js, React, Tailwind CSS, Framer Motion, and Three.js. Focused on crafting ultra-premium, interactive user experiences and dynamic data visualization.",
            "fullstack": "Versatile Senior Full-Stack Engineer with end-to-end expertise in Next.js, Node.js, MongoDB, PyTorch ML microservices, and vector search systems. Proven track record of building production systems with modern UI/UX."
        }

    def get_targeted_summary(self, role: str = "fullstack") -> dict:
        clean_role = (role or "fullstack").lower()
        if clean_role not in self.role_summaries:
            clean_role = "fullstack"

        return {
            "role": clean_role,
            "targetedSummary": self.role_summaries[clean_role]
        }

    def rank_skills(self, skills: list, role: str = "fullstack") -> list:
        clean_role = (role or "fullstack").lower()
        
        keywords = {
            "aiml": ["python", "pytorch", "tensorflow", "ml", "ai", "nlp", "onnx", "vector", "scikit-learn", "data"],
            "backend": ["node", "express", "mongodb", "mongoose", "docker", "sql", "api", "rest", "backend", "jwt"],
            "frontend": ["react", "next", "tailwind", "css", "three", "framer", "ui", "ux", "frontend", "html", "javascript", "typescript"]
        }.get(clean_role, [])

        ranked = []
        for s in skills:
            s_copy = dict(s) if isinstance(s, dict) else {"name": str(s)}
            name = str(s_copy.get("name", s_copy.get("title", ""))).lower()
            
            score = 80
            for kw in keywords:
             if kw in name:
              score += 15

            s_copy["mlRoleScore"] = min(99, score)
            ranked.append(s_copy)

        ranked.sort(key=lambda x: x["mlRoleScore"], reverse=True)
        return ranked

    def rank_internships(self, internships: list, role: str = "fullstack") -> list:
        clean_role = (role or "fullstack").lower()
        
        keywords = {
            "aiml": ["ai", "ml", "machine learning", "python", "pytorch", "model", "data", "algorithm"],
            "backend": ["backend", "api", "node", "database", "express", "mongodb", "server", "microservice"],
            "frontend": ["frontend", "react", "ui", "ux", "design", "next.js", "tailwind", "component"]
        }.get(clean_role, [])

        ranked = []
        for item in internships:
            item_copy = dict(item) if isinstance(item, dict) else {"title": str(item)}
            title = str(item_copy.get("title", "")).lower()
            role_desc = str(item_copy.get("role", "")).lower()
            desc = str(item_copy.get("description", "")).lower()
            full_str = f"{title} {role_desc} {desc}"

            score = 82
            for kw in keywords:
             if kw in full_str:
              score += 10

            item_copy["mlRoleScore"] = min(99, score)
            ranked.append(item_copy)

        ranked.sort(key=lambda x: x["mlRoleScore"], reverse=True)
        return ranked

resume_skills_engine = ResumeSkillsPersonalizer()
