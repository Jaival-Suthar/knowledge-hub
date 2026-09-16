import json
import re
from pathlib import Path

M1 = Path("evaluation/datasets/questions-gold-evidence-v1.json")
M2 = Path("evaluation/datasets/m2_chunks.json")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


questions = json.loads(M1.read_text(encoding="utf-8"))
chunks = json.loads(M2.read_text(encoding="utf-8"))

total = 0
matched = 0
unresolved = 0
ambiguous = 0

for question in questions:
    evidence = question.get("gold_evidence", []) + question.get(
        "acceptable_evidence", []
    )

    for index, item in enumerate(evidence):
        total += 1

        page = item["page_number"]
        span = normalize(item["text_span"])

        hits = [
            chunk
            for chunk in chunks
            if chunk["page"] == page and span in normalize(chunk["content"])
        ]

        if len(hits) == 1:
            matched += 1
            print(
                f"OK         {question['id']} "
                f"{item['relevance']:10} "
                f"grade={item['grade']} "
                f"page={page} "
                f"M2_INDEX={hits[0]['index']} "
                f"M2_ID={hits[0]['chunk_id']}"
            )

        elif len(hits) == 0:
            unresolved += 1
            print(
                f"UNRESOLVED {question['id']} "
                f"{item['relevance']:10} "
                f"grade={item['grade']} "
                f"page={page}"
            )
            print(f"  span: {item['text_span']}")

        else:
            ambiguous += 1
            print(
                f"AMBIGUOUS  {question['id']} "
                f"{item['relevance']:10} "
                f"grade={item['grade']} "
                f"page={page} "
                f"matches={len(hits)}"
            )
            for hit in hits:
                print(f"  M2_INDEX={hit['index']} M2_ID={hit['chunk_id']}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total evidence records : {total}")
print(f"Exact single-chunk     : {matched}")
print(f"Unresolved             : {unresolved}")
print(f"Ambiguous              : {ambiguous}")
