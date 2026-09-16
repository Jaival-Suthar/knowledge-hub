import json
import re
from pathlib import Path

M1 = Path("evaluation/datasets/questions-gold-evidence-v1.json")
M2 = Path("evaluation/datasets/m2_chunks.json")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def fragments(text: str) -> list[str]:
    normalized = normalize(text)
    words = normalized.split()

    if len(words) <= 8:
        return [normalized]

    # Use beginning, middle, and ending distinctive fragments.
    return [
        " ".join(words[:8]),
        " ".join(words[len(words) // 2 - 4 : len(words) // 2 + 4]),
        " ".join(words[-8:]),
    ]


questions = json.loads(M1.read_text(encoding="utf-8"))
chunks = json.loads(M2.read_text(encoding="utf-8"))

for question in questions:
    evidence = question.get("gold_evidence", []) + question.get(
        "acceptable_evidence", []
    )

    for ev_index, ev in enumerate(evidence):
        page = ev["page_number"]
        span = ev["text_span"]

        page_chunks = [c for c in chunks if c["page"] == page]

        span_norm = normalize(span)

        exact = [c for c in page_chunks if span_norm in normalize(c["content"])]

        if exact:
            continue

        scores = []

        for chunk in page_chunks:
            content = normalize(chunk["content"])
            fragment_hits = sum(fragment in content for fragment in fragments(span))

            if fragment_hits:
                scores.append((fragment_hits, chunk))

        scores.sort(
            key=lambda item: (
                -item[0],
                item[1]["index"],
            )
        )

        print()
        print("=" * 90)
        print(
            f"{question['id']} | {ev.get('relevance')} | "
            f"grade={ev.get('grade')} | page={page}"
        )
        print("M1 SPAN:")
        print(span)
        print()

        if not scores:
            print("NO FRAGMENT CANDIDATES")
            continue

        print("M2 CANDIDATES:")

        for score, chunk in scores[:5]:
            print()
            print(f"  score={score} index={chunk['index']} id={chunk['chunk_id']}")
            print(f"  content={chunk['content'][:1500]}")
