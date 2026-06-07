import json

with open("data/candidates.jsonl","r",encoding="utf-8") as f:
    line=f.readline()

candidate=json.loads(line)

print(candidate.keys())
print(candidate)