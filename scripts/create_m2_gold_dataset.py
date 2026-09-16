import json
from pathlib import Path

SOURCE = Path("evaluation/datasets/questions-gold-evidence-v1.json")
MAPPING = Path("evaluation/datasets/m2_gold_mapping-v1.json")
OUTPUT = Path("evaluation/datasets/m2_gold-evidence-v1.json")

questions = json.loads(SOURCE.read_text(encoding="utf-8"))
mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

by_key = {
    (r["question_id"], r["evidence_type"], r["m1_chunk_id"]): r
    for r in mapping["records"]
}

output = []

for question in questions:
    item = dict(question)
    evidence = []

    for evidence_type in ("primary", "acceptable"):
        for record in (
            question.get("gold_evidence", [])
            if evidence_type == "primary"
            else question.get("acceptable_evidence", [])
        ):
            key = (
                question["id"],
                evidence_type,
                record["chunk_id"],
            )

            mapped = by_key[key]

            if len(mapped["m2_chunk_ids"]) != 1:
                raise ValueError(
                    f"Expected exactly one M2 chunk for {key}, "
                    f"got {mapped['m2_chunk_ids']}"
                )

            evidence.append(
                {
                    **record,
                    "m1_chunk_id": record["chunk_id"],
                    "m2_chunk_id": mapped["m2_chunk_ids"][0],
                    "mapping_status": mapped["status"],
                }
            )

    item["m2_gold_evidence"] = evidence
    output.append(item)

OUTPUT.write_text(
    json.dumps(output, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"Wrote: {OUTPUT}")
print(f"Questions: {len(output)}")
print(f"Evidence records: {sum(len(q['m2_gold_evidence']) for q in output)}")
print(
    "Unanswerable:",
    sum(not q["answerable"] for q in output),
)
