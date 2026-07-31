#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_INPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
DEFAULT_PERSIST_DIR = BASE_DIR / "chroma_db"

DEFAULT_COLLECTION_NAME = "kazemachi_game_knowledge"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 64


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 chunks 檔案：{path}")

    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失敗: {e}") from e

            if not isinstance(item, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是 JSON object")
            chunks.append(item)

    if not chunks:
        raise ValueError("chunks.jsonl 是空的，沒有資料可建立 embeddings。")

    return chunks


def build_document_text(chunk: Dict[str, Any]) -> str:
    title = str(chunk.get("title", "")).strip()
    section = str(chunk.get("section", "")).strip()
    text = str(chunk.get("text", "")).strip()
    summary = str(chunk.get("summary", "")).strip()

    parts: List[str] = []

    if title:
        parts.append(f"標題：{title}")
    if section:
        parts.append(f"段落：{section}")
    if summary:
        parts.append(f"摘要：{summary}")
    if text:
        parts.append(f"內容：{text}")

    merged = "\n".join(parts).strip()
    if not merged:
        raise ValueError(f"chunk `{chunk.get('chunk_id', 'unknown')}` 沒有可用於 embedding 的文字。")
    return merged


def sanitize_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def sanitize_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    for key, value in chunk.items():
        if key == "text":
            continue
        metadata[key] = sanitize_metadata_value(value)

    return metadata


def prepare_records(chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    seen_ids = set()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            raise ValueError("某筆 chunk 缺少 `chunk_id`。")

        if chunk_id in seen_ids:
            raise ValueError(f"發現重複的 chunk_id：{chunk_id}")
        seen_ids.add(chunk_id)

        doc_text = build_document_text(chunk)
        metadata = sanitize_metadata(chunk)

        ids.append(chunk_id)
        documents.append(doc_text)
        metadatas.append(metadata)

    return ids, documents, metadatas


def batched(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def get_embedding_function(model_name: str):
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    model_name: str,
):
    embedding_function = get_embedding_function(model_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={
            "description": "Kazemachi village game RAG knowledge base",
            "embedding_model": model_name
        }
    )
    return collection


def reset_collection_if_needed(client: chromadb.PersistentClient, collection_name: str, reset: bool):
    if not reset:
        return

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"已刪除既有 collection：{collection_name}")


def main():
    parser = argparse.ArgumentParser(description="Build local embeddings with Sentence Transformers + Chroma.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_FILE), help="Path to chunks.jsonl")
    parser.add_argument("--persist-dir", type=str, default=str(DEFAULT_PERSIST_DIR), help="Chroma persist directory")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME, help="Collection name")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Sentence Transformers model name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for inserts")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection before insert")
    args = parser.parse_args()

    input_path = Path(args.input)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"讀取 chunks：{input_path}")
    chunks = load_chunks(input_path)
    print(f"載入 {len(chunks)} 筆 chunks")

    ids, documents, metadatas = prepare_records(chunks)

    print(f"初始化 Chroma PersistentClient：{persist_dir}")
    client = chromadb.PersistentClient(path=str(persist_dir))

    reset_collection_if_needed(client, args.collection, args.reset)

    print(f"載入 embedding model：{args.model}")
    collection = get_or_create_collection(
        client=client,
        collection_name=args.collection,
        model_name=args.model,
    )

    existing_ids = set(collection.get(include=[])["ids"])
    new_records = [
        (cid, doc, meta)
        for cid, doc, meta in zip(ids, documents, metadatas)
        if cid not in existing_ids
    ]

    if not new_records:
        print("沒有新資料可寫入，所有 chunk_id 已存在。")
        print(f"collection: {args.collection}")
        print(f"persist dir: {persist_dir}")
        return

    print(f"準備新增 {len(new_records)} 筆資料到 collection `{args.collection}`")

    batches = batched(new_records, args.batch_size)
    total = len(new_records)
    inserted = 0

    for batch_index, batch in enumerate(batches, start=1):
        batch_ids = [x[0] for x in batch]
        batch_docs = [x[1] for x in batch]
        batch_meta = [x[2] for x in batch]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )

        inserted += len(batch)
        print(f"[{batch_index}/{len(batches)}] 已寫入 {inserted}/{total}")

    final_count = collection.count()
    print("完成。")
    print(f"- collection: {args.collection}")
    print(f"- persist dir: {persist_dir}")
    print(f"- embedding model: {args.model}")
    print(f"- collection count: {final_count}")


if __name__ == "__main__":
    main()