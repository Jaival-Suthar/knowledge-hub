from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

GOLD_DATASET = Path("evaluation/datasets/questions-gold-evidence-v1.json")
M2_CHUNKS = Path("evaluation/datasets/m2_chunks.json")

OUTPUT_MAPPING = Path("evaluation/datasets/m2_gold_mapping-v1.json")
OUTPUT_REPORT = Path("evaluation/reports/m2_gold_mapping-v1.txt")


# These are human-verified resolutions for legitimate overlapping M2 chunks.
# The automatic matcher correctly found both chunks, but manual inspection
# established which chunk is the canonical evidence location.
MANUAL_RESOLUTIONS = {
    ("Q003", "primary", 47): [
        "785b31676f422035f504c4a649c3673e38e95375c258c84d0c80f34ba6196aa0"
    ],
    ("Q003", "primary", 50): [
        "25b74d8908aba01a564ec4a5350d1cfbedd444d432d8d1164d022220ed73dbf1"
    ],
    ("Q004", "primary", 50): [
        "25b74d8908aba01a564ec4a5350d1cfbedd444d432d8d1164d022220ed73dbf1"
    ],
    ("Q010", "acceptable", 65): [
        "d5038834ea61f21f06e450aaa25e316b172ee2494b6e70b8130e698490787786"
    ],
    ("Q021", "primary", 47): [
        "785b31676f422035f504c4a649c3673e38e95375c258c84d0c80f34ba6196aa0"
    ],
    ("Q021", "primary", 50): [
        "25b74d8908aba01a564ec4a5350d1cfbedd444d432d8d1164d022220ed73dbf1"
    ],
    ("Q026", "primary", 50): [
        "25b74d8908aba01a564ec4a5350d1cfbedd444d432d8d1164d022220ed73dbf1"
    ],
}


def normalize(text: str) -> str:
    """Normalize whitespace and punctuation enough for evidence comparison."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def compact(text: str) -> str:
    """More aggressive normalization for PDF extraction differences."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_matches(
    evidence: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find chunks containing the complete normalized evidence span."""
    target = normalize(evidence["text_span"])
    page = evidence["page_number"]

    matches = []

    for chunk in chunks:
        if chunk["page"] != page:
            continue

        content = normalize(chunk["content"])

        if target and target in content:
            matches.append(chunk)

    return matches


def compact_matches(
    evidence: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Find chunks containing the evidence after aggressive normalization.

    This handles PDF extraction differences such as:
      "incorrectly identifythe problem"
      vs
      "incorrectly identify the problem"
    """
    target = compact(evidence["text_span"])
    page = evidence["page_number"]

    if not target:
        return []

    matches = []

    for chunk in chunks:
        if chunk["page"] != page:
            continue

        content = compact(chunk["content"])

        if target in content:
            matches.append(chunk)

    return matches


def token_fragments(text: str, size: int = 8) -> list[str]:
    """Generate distinctive token fragments from an evidence span."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.casefold())

    if len(tokens) <= size:
        return [" ".join(tokens)]

    fragments = []

    positions = {
        0,
        max(0, len(tokens) // 2 - size // 2),
        max(0, len(tokens) - size),
    }

    for start in sorted(positions):
        fragment = " ".join(tokens[start : start + size])

        if fragment:
            fragments.append(fragment)

    return fragments


def fragment_matches(
    evidence: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Find page-local chunks containing distinctive fragments.

    This is candidate discovery only. It is never automatically accepted.
    """
    page = evidence["page_number"]
    fragments = token_fragments(evidence["text_span"])

    candidates: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        if chunk["page"] != page:
            continue

        content = normalize(chunk["content"])

        hits = sum(1 for fragment in fragments if fragment and fragment in content)

        if hits:
            candidates[chunk["chunk_id"]] = {
                **chunk,
                "_fragment_hits": hits,
            }

    return sorted(
        candidates.values(),
        key=lambda item: (
            -item["_fragment_hits"],
            item["index"],
        ),
    )


def classify(
    evidence: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Classify an evidence record.

    EXACT:
        Normalized full-span match in exactly one chunk.

    AMBIGUOUS:
        Full-span match appears in multiple chunks.

    CANDIDATE:
        No exact match, but fragment discovery finds candidates.

    UNRESOLVED:
        No useful candidate found.
    """
    matches = exact_matches(evidence, chunks)

    if len(matches) == 1:
        return "exact", matches

    if len(matches) > 1:
        return "ambiguous", matches

    compact_result = compact_matches(evidence, chunks)

    if len(compact_result) == 1:
        return "exact", compact_result

    if len(compact_result) > 1:
        return "ambiguous", compact_result

    candidates = fragment_matches(evidence, chunks)

    if candidates:
        return "candidate", candidates

    return "unresolved", []


def evidence_records(dataset: list[dict[str, Any]]):
    """Flatten primary and acceptable evidence while preserving provenance."""
    for question in dataset:
        for evidence_type in ("gold_evidence", "acceptable_evidence"):
            for evidence in question.get(evidence_type, []):
                yield question, evidence_type, evidence


def build_mapping(
    dataset: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    records = []
    report_lines: list[str] = []

    counts = {
        "exact": 0,
        "manual_resolution": 0,
        "candidate": 0,
        "ambiguous": 0,
        "unresolved": 0,
    }

    for question, evidence_type, evidence in evidence_records(dataset):
        status, matches = classify(evidence, chunks)

        evidence_type_name = (
            "primary" if evidence_type == "gold_evidence" else "acceptable"
        )

        resolution_key = (
            question["id"],
            evidence_type_name,
            evidence["page_number"],
        )

        if status == "ambiguous" and resolution_key in MANUAL_RESOLUTIONS:
            resolved_ids = set(MANUAL_RESOLUTIONS[resolution_key])

            matches = [match for match in matches if match["chunk_id"] in resolved_ids]

            status = "manual_resolution"

        counts[status] += 1

        candidate_ids = [match["chunk_id"] for match in matches]

        record = {
            "question_id": question["id"],
            "evidence_type": evidence_type_name,
            "grade": evidence["grade"],
            "relevance": evidence["relevance"],
            "m1_page": evidence["page_number"],
            "m1_chunk_id": evidence["chunk_id"],
            "m1_text_span": evidence["text_span"],
            "m2_chunk_ids": candidate_ids,
            "status": status,
        }

        records.append(record)

        report_lines.extend(
            [
                "=" * 70,
                (
                    f"{question['id']} | "
                    f"{record['evidence_type']} | "
                    f"grade={record['grade']} | "
                    f"page={record['m1_page']}"
                ),
                "=" * 70,
                "",
                "M1 evidence:",
                evidence["text_span"],
                "",
                f"STATUS: {status.upper()}",
            ]
        )

        if matches:
            report_lines.extend(
                [
                    "",
                    "M2 candidates:",
                ]
            )

            for match in matches:
                report_lines.extend(
                    [
                        "",
                        f"  index : {match['index']}",
                        f"  page  : {match['page']}",
                        f"  id    : {match['chunk_id']}",
                        "  content:",
                        f"  {match['content']}",
                    ]
                )

        report_lines.append("")

    summary = [
        "",
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        f"Total evidence records : {sum(counts.values())}",
        f"Exact                  : {counts['exact']}",
        f"Manual resolution      : {counts['manual_resolution']}",
        f"Candidate              : {counts['candidate']}",
        f"Ambiguous              : {counts['ambiguous']}",
        f"Unresolved             : {counts['unresolved']}",
    ]

    report_lines.extend(summary)

    mapping = {
        "dataset_version": "m2-gold-mapping-v1",
        "source_dataset": str(GOLD_DATASET),
        "document": {
            "name": "The 10X Rule.pdf",
            "m2_chunk_count": len(chunks),
        },
        "records": records,
        "summary": counts,
    }

    return mapping, "\n".join(report_lines) + "\n"


def main() -> int:
    dataset = load_json(GOLD_DATASET)
    chunks = load_json(M2_CHUNKS)

    mapping, report = build_mapping(dataset, chunks)

    OUTPUT_MAPPING.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_MAPPING.write_text(
        json.dumps(
            mapping,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    OUTPUT_REPORT.write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print(f"\nWrote mapping: {OUTPUT_MAPPING}")

    print(f"Wrote report : {OUTPUT_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
