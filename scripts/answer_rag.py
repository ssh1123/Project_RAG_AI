#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = BASE_DIR / "chroma_db"
DEFAULT_COLLECTION_NAME = "kazemachi_game_knowledge"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 4


def get_embedding_function(model_name: str):
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_collection(
    persist_dir: Path,
    collection_name: str,
    model_name: str,
):
    client = chromadb.PersistentClient(path=str(persist_dir))
    embedding_function = get_embedding_function(model_name)
    return client.get_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )


def parse_where_filters(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    where: Dict[str, Any] = {}

    if args.doc_type:
        where["document_type"] = args.doc_type

    if args.chapter_gate is not None:
        where["chapter_gate"] = str(args.chapter_gate)

    if args.spoiler_level:
        where["spoiler_level"] = args.spoiler_level

    return where if where else None


def query_collection(
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    query_text: str,
    top_k: int,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    collection = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
    )

    kwargs = {
        "query_texts": [query_text],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def build_context_blocks(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    distances: List[Any],
) -> List[Dict[str, Any]]:
    blocks = []

    for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        block = {
            "rank": idx,
            "chunk_id": meta.get("chunk_id", ""),
            "document_id": meta.get("document_id", ""),
            "document_type": meta.get("document_type", ""),
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "source_file": meta.get("source_file", ""),
            "chapter_gate": meta.get("chapter_gate", ""),
            "spoiler_level": meta.get("spoiler_level", ""),
            "distance": dist,
            "text": doc,
        }
        blocks.append(block)

    return blocks


def format_context_for_prompt(context_blocks: List[Dict[str, Any]]) -> str:
    parts = []

    for block in context_blocks:
        part = (
            f"[來源 {block['rank']}]\n"
            f"chunk_id: {block['chunk_id']}\n"
            f"document_id: {block['document_id']}\n"
            f"document_type: {block['document_type']}\n"
            f"title: {block['title']}\n"
            f"section: {block['section']}\n"
            f"source_file: {block['source_file']}\n"
            f"distance: {block['distance']}\n"
            f"text:\n{block['text']}"
        )
        parts.append(part)

    return "\n\n---\n\n".join(parts)


def build_system_prompt() -> str:
    return (
        "你是《風待村》的遊戲內導覽 AI。\n"
        "你的任務是根據提供的檢索內容，回答玩家關於遊戲世界觀、地方創生概念、地點、角色與任務背景的問題。\n\n"
        "回答規則：\n"
        "1. 只能根據提供的 context 回答，不可自行補完未出現在 context 的事實。\n"
        "2. 如果 context 不足以回答，必須明確說「目前資料不足以回答這個問題」。\n"
        "3. 回答要優先使用《風待村》世界觀語氣，但內容仍要清楚、可教學。\n"
        "4. 若適合，可先用一句簡短答案，再補充解釋。\n"
        "5. 回答最後附上引用來源，格式使用 [來源1]、[來源2]。\n"
        "6. 不可透露未解鎖劇情，不可超出 chapter_gate 限制。"
    )


def build_user_prompt(user_question: str, context_text: str) -> str:
    return (
        f"玩家問題：{user_question}\n\n"
        f"以下是可用的檢索資料，請只根據這些資料回答：\n\n"
        f"{context_text}\n\n"
        "請輸出：\n"
        "1. 簡短回答\n"
        "2. 詳細說明\n"
        "3. 引用來源"
    )


def build_answer_skeleton(
    user_question: str,
    context_blocks: List[Dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    citations = [
        {
            "label": f"來源{block['rank']}",
            "chunk_id": block["chunk_id"],
            "document_id": block["document_id"],
            "title": block["title"],
            "section": block["section"],
            "source_file": block["source_file"],
        }
        for block in context_blocks
    ]

    return {
        "question": user_question,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "retrieved_context": context_blocks,
        "expected_answer_format": {
            "short_answer": "",
            "detailed_answer": "",
            "citations": ["[來源1]", "[來源2]"]
        },
        "citations": citations
    }


def main():
    parser = argparse.ArgumentParser(description="Build RAG answer skeleton from local Chroma retrieval.")
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("--persist-dir", type=str, default=str(DEFAULT_PERSIST_DIR), help="Chroma persist directory")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME, help="Collection name")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Sentence Transformers model name")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--doc-type", type=str, choices=["concept", "lore"], help="Filter by document_type")
    parser.add_argument("--chapter-gate", type=int, help="Filter by chapter_gate")
    parser.add_argument("--spoiler-level", type=str, help="Filter by spoiler_level")
    parser.add_argument("--context-only", action="store_true", help="Print prompt-ready context only")
    parser.add_argument("--prompt-only", action="store_true", help="Print system prompt + user prompt only")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    where = parse_where_filters(args)

    result = query_collection(
        persist_dir=persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        query_text=args.query,
        top_k=args.top_k,
        where=where,
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    if not documents:
        output = {
            "question": args.query,
            "error": "查無相關 context",
            "system_prompt": build_system_prompt(),
            "user_prompt": "",
            "retrieved_context": [],
            "citations": []
        }
        print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    context_blocks = build_context_blocks(documents, metadatas, distances)
    context_text = format_context_for_prompt(context_blocks)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(args.query, context_text)

    if args.context_only:
        print(context_text)
        return

    if args.prompt_only:
        print("=== SYSTEM PROMPT ===\n")
        print(system_prompt)
        print("\n=== USER PROMPT ===\n")
        print(user_prompt)
        return

    output = build_answer_skeleton(
        user_question=args.query,
        context_blocks=context_blocks,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()