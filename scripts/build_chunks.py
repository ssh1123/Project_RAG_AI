#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONCEPTS_DIR = DATA_DIR / "concepts"
LORE_DIR = DATA_DIR / "lore"
NPC_DIR = DATA_DIR / "npc"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

MAX_CHARS = 700
MIN_CHARS = 120

REQUIRED_FIELDS = [
    "id",
    "type",
    "title",
    "summary",
    "chapter_gate",
    "spoiler_level",
    "tags",
    "content",
]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是單一 JSON object。")
    return data


def validate_document(doc: Dict[str, Any], path: Path) -> List[str]:
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in doc:
            errors.append(f"{path.name}: 缺少欄位 `{field}`")

    if "type" in doc and doc["type"] not in {"concept", "lore", "npc"}:
        errors.append(f"{path.name}: `type` 必須是 concept 或 lore 或 npc")

    for list_field in [
        "related_concepts",
        "keywords",
        "suggested_questions",
        "location_id",
        "npc_id",
        "quest_id",
        "tags",
    ]:
        if list_field in doc and not isinstance(doc[list_field], list):
            errors.append(f"{path.name}: `{list_field}` 必須是 list")

    if "chapter_gate" in doc and not isinstance(doc["chapter_gate"], int):
        errors.append(f"{path.name}: `chapter_gate` 必須是 int")

    if "content" in doc and not isinstance(doc["content"], str):
        errors.append(f"{path.name}: `content` 必須是 string")

    return errors


def split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def split_long_paragraph(paragraph: str, max_chars: int = MAX_CHARS) -> List[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[。！？!?])\s*", paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else current + " " + sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


def merge_short_chunks(chunks: List[str], min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> List[str]:
    if not chunks:
        return []

    merged = []
    buffer = ""

    for chunk in chunks:
        if not buffer:
            buffer = chunk
            continue

        if len(buffer) < min_chars and len(buffer) + 2 + len(chunk) <= max_chars:
            buffer = buffer + "\n\n" + chunk
        else:
            merged.append(buffer.strip())
            buffer = chunk

    if buffer:
        if merged and len(buffer) < min_chars and len(merged[-1]) + 2 + len(buffer) <= max_chars:
            merged[-1] = merged[-1] + "\n\n" + buffer
        else:
            merged.append(buffer.strip())

    return merged


def chunk_text(text: str) -> List[str]:
    paragraphs = split_paragraphs(normalize_text(text))
    stage_1 = []

    for paragraph in paragraphs:
        stage_1.extend(split_long_paragraph(paragraph, max_chars=MAX_CHARS))

    final_chunks = merge_short_chunks(stage_1, min_chars=MIN_CHARS, max_chars=MAX_CHARS)
    return final_chunks


def extract_section_label(chunk_text: str) -> str:
    match = re.match(r"^\s*([^：:\n]{1,30})[：:]", chunk_text)
    if match:
        return match.group(1).strip()
    return ""


def build_chunk_record(
    doc: Dict[str, Any],
    source_file: Path,
    chunk_text: str,
    chunk_index: int,
) -> Dict[str, Any]:
    chunk_id = f"{doc['id']}_c{chunk_index:02d}"
    section = extract_section_label(chunk_text)

    record = {
        "chunk_id": chunk_id,
        "document_id": doc.get("id"),
        "document_type": doc.get("type"),
        "title": doc.get("title"),
        "summary": doc.get("summary", ""),
        "section": section,
        "text": chunk_text,
        "related_concepts": doc.get("related_concepts", []),
        "keywords": doc.get("keywords", []),
        "suggested_questions": doc.get("suggested_questions", []),
        "location_id": doc.get("location_id", []),
        "npc_id": doc.get("npc_id", []),
        "quest_id": doc.get("quest_id", []),
        "chapter_gate": doc.get("chapter_gate"),
        "spoiler_level": doc.get("spoiler_level"),
        "difficulty": doc.get("difficulty", None),
        "tags": doc.get("tags", []),
        "source_file": str(source_file.relative_to(BASE_DIR)).replace("\\", "/"),
        "char_count": len(chunk_text),
    }

    optional_fields = [
        "definition",
        "importance",
        "town_example",
        "misconception",
        "time_period",
        "core_issue",
        "conflict",
    ]
    for field in optional_fields:
        if field in doc:
            record[field] = doc[field]

    return record


def collect_documents() -> Tuple[List[Tuple[Dict[str, Any], Path]], List[str]]:
    documents = []
    errors = []

    for folder in [CONCEPTS_DIR, LORE_DIR, NPC_DIR]:
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.json")):
            try:
                doc = load_json_file(path)
                validation_errors = validate_document(doc, path)
                if validation_errors:
                    errors.extend(validation_errors)
                    continue
                documents.append((doc, path))
            except Exception as e:
                errors.append(f"{path.name}: 讀取失敗 - {e}")

    return documents, errors


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents, errors = collect_documents()

    if errors:
        print("文件驗證失敗：")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    all_chunks = []
    seen_chunk_ids = set()

    for doc, path in documents:
        chunks = chunk_text(doc["content"])

        for idx, chunk in enumerate(chunks, start=1):
            record = build_chunk_record(doc, path, chunk, idx)

            if record["chunk_id"] in seen_chunk_ids:
                raise ValueError(f"重複的 chunk_id: {record['chunk_id']}")

            seen_chunk_ids.add(record["chunk_id"])
            all_chunks.append(record)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    stats = {}
    for chunk in all_chunks:
        doc_type = chunk["document_type"]
        stats[doc_type] = stats.get(doc_type, 0) + 1

    print(f"完成：輸出 {len(all_chunks)} 筆 chunks 到 {OUTPUT_FILE}")
    for doc_type, count in sorted(stats.items()):
        print(f"- {doc_type}: {count} chunks")


if __name__ == "__main__":
    main()