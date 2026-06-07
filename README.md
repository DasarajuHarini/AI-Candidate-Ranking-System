# AI Candidate Ranking System

## Problem Statement

Build an AI-powered candidate ranking system that understands job requirements and ranks candidates based on overall fit rather than simple keyword matching.

## Approach

### Semantic Matching

Used Sentence Transformers (all-MiniLM-L6-v2) to generate embeddings for:

* Job Description
* Candidate Profile
* Career History

Cosine similarity is used to calculate semantic relevance.

### Skills Matching

Matched important AI skills such as:

* Python
* NLP
* LLM
* LoRA
* Retrieval
* Ranking
* Milvus
* FAISS

### Experience Scoring

Candidates with 5–9 years of experience receive higher scores.

### Behavioral Signals

Used Redrob platform signals:

* Open To Work
* Recruiter Response Rate
* GitHub Activity
* Notice Period
* Relocation Preference

### Final Score

Final Score =
50% Semantic Match +
20% Skills Match +
15% Experience Match +
15% Behavioral Signals

## Technologies Used

* Python
* Pandas
* Sentence Transformers
* Scikit-Learn

## Output

* ranked_candidates.csv
* sample_submission.csv

## Future Improvements

* Cross Encoder Re-ranking
* LLM-based reasoning
* Dynamic skill extraction
* Vector database integration
