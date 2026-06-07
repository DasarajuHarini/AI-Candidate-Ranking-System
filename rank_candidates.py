import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading AI model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

job_text = """
Senior AI Engineer
Embeddings
Retrieval Systems
Ranking Systems
LLMs
Fine Tuning
Vector Databases
Python
Evaluation Frameworks
Recommendation Systems
Search Systems
"""

jd_embedding = model.encode(job_text)

required_skills = [
    "Python",
    "Embeddings",
    "Milvus",
    "Pinecone",
    "FAISS",
    "LLM",
    "LoRA",
    "NLP",
    "Retrieval",
    "Ranking"
]

results = []

print("Processing candidates...")

with open("data/candidates.jsonl", "r", encoding="utf-8", errors="ignore") as f:

    for i, line in enumerate(f):

        # Testing limit
        if i >= 20000:
            break

        try:
            candidate = json.loads(line)

            if i % 500 == 0:
                print(f"Processed {i} candidates...")

            profile = candidate.get("profile", {})

            # Career history text
            career_text = ""
            for job in candidate.get("career_history", []):
                career_text += " " + job.get("description", "")

            candidate_text = (
                profile.get("headline", "") + " " +
                profile.get("summary", "") + " " +
                profile.get("current_title", "") + " " +
                career_text
            )

            candidate_embedding = model.encode(candidate_text)

            semantic_score = cosine_similarity(
                [jd_embedding],
                [candidate_embedding]
            )[0][0]

            # Experience Score
            exp = profile.get("years_of_experience", 0)

            if 5 <= exp <= 9:
                exp_score = 1.0
            elif 4 <= exp < 5:
                exp_score = 0.8
            elif 9 < exp <= 12:
                exp_score = 0.7
            else:
                exp_score = 0.3

            # Skills Score
            skills = [
                s.get("name", "").lower()
                for s in candidate.get("skills", [])
            ]

            matched = 0

            for skill in required_skills:
                if skill.lower() in skills:
                    matched += 1

            skills_score = matched / len(required_skills)

            # Behavioral Score
            signals = candidate.get("redrob_signals", {})

            behavior_score = 0

            if signals.get("open_to_work_flag"):
                behavior_score += 1

            if signals.get("willing_to_relocate"):
                behavior_score += 1

            if signals.get("notice_period_days", 999) <= 30:
                behavior_score += 1

            if signals.get("recruiter_response_rate", 0) > 0.6:
                behavior_score += 1

            if signals.get("github_activity_score", 0) > 40:
                behavior_score += 1

            behavior_score = behavior_score / 5

            # Final Score
            final_score = (
                0.50 * semantic_score +
                0.15 * exp_score +
                0.20 * skills_score +
                0.15 * behavior_score
            )

            results.append({
                "candidate_id": candidate["candidate_id"],
                "semantic_score": round(float(semantic_score), 4),
                "experience_score": round(float(exp_score), 4),
                "skills_score": round(float(skills_score), 4),
                "behavior_score": round(float(behavior_score), 4),
                "final_score": round(float(final_score), 4)
            })

        except Exception as e:
            print("ERROR:", e)
            continue

# Ranked Output
ranked = pd.DataFrame(results)

ranked = ranked.sort_values(
    by="final_score",
    ascending=False
)

# Save detailed ranking
ranked.to_csv(
    "output/ranked_candidates.csv",
    index=False
)

# Save submission file
submission = ranked[["candidate_id", "final_score"]].copy()

submission["rank"] = range(
    1,
    len(submission) + 1
)

submission = submission[
    ["candidate_id", "rank", "final_score"]
]

submission.columns = [
    "candidate_id",
    "rank",
    "score"
]

submission.to_csv(
    "output/sample_submission.csv",
    index=False
)

print("Submission file created!")

print("\nDone!")
print("Candidates Processed:", len(ranked))
print("\nTop 10 Candidates:")
print(ranked.head(10))