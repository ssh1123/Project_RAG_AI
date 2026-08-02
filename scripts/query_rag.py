#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# 改成使用 simple_vector_store，而不是 chromadb
from simple_vector_store import load_index, search

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TOP_K = 4


def shorten(text: str, max_len: int = 220) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_result_block(
    rank: int,
    doc: str,
    metadata: Dict[str, Any],
    distance: Any,
) -> None:
    print("=" * 80)
    print(f"[{rank}] chunk_id      : {metadata.get('chunk_id', '')}")
    print(f"     document_id   : {metadata.get('document_id', '')}")
    print(f"     document_type : {metadata.get('document_type', '')}")
    print(f"     title         : {metadata.get('title', '')}")
    print(f"     section       : {metadata.get('section', '')}")
    print(f"     source_file   : {metadata.get('source_file', '')}")
    print(f"     chapter_gate  : {metadata.get('chapter_gate', '')}")
    print(f"     spoiler_level : {metadata.get('spoiler_level', '')}")
    print(f"     distance      : {distance}")
    print(f"     tags          : {metadata.get('tags', '')}")
    print(f"     related       : {metadata.get('related_concepts', '')}")
    print("-" * 80)
    print(shorten(doc, max_len=500))
    print()


def format_context_for_llm(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    distances: List[Any],
) -> str:
    blocks = []

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        block = (
            f"[來源 {i}]\n"
            f"chunk_id: {meta.get('chunk_id', '')}\n"
            f"document_id: {meta.get('document_id', '')}\n"
            f"title: {meta.get('title', '')}\n"
            f"section: {meta.get('section', '')}\n"
            f"distance: {dist}\n"
            f"text:\n{doc}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query local simple_vector_store index (no Chroma)."
    )
    parser.add_argument("query", type=str, help="User query text")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON-like result (documents + metadatas + distances)",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Print prompt-ready retrieved context only",
    )
    args = parser.parse_args()

    # 啟動時載入 simple_index.npz 與 embedding 模型
    load_index()

    print(f"查詢問題：{args.query}")
    print(f"top_k：{args.top_k}")
    print()

    documents, metadatas, distances = search(
        query=args.query,
        top_k=args.top_k,
    )

    if not documents:
        print("查無結果。")
        return

    if args.json:
        result = {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.context_only:
        context = format_context_for_llm(documents, metadatas, distances)
        print(context)
        return

    print("檢索結果：\n")
    for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print_result_block(
            rank=idx,
            doc=doc,
            metadata=meta,
            distance=dist,
        )


if __name__ == "__main__":
    main()