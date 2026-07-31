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

    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )
    return collection


def parse_where_filters(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    where: Dict[str, Any] = {}

    if args.doc_type:
        where["document_type"] = args.doc_type

    if args.chapter_gate is not None:
        where["chapter_gate"] = str(args.chapter_gate)

    if args.spoiler_level:
        where["spoiler_level"] = args.spoiler_level

    return where if where else None


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
):
    print("=" * 80)
    print(f"[{rank}] chunk_id       : {metadata.get('chunk_id', '')}")
    print(f"    document_id    : {metadata.get('document_id', '')}")
    print(f"    document_type  : {metadata.get('document_type', '')}")
    print(f"    title          : {metadata.get('title', '')}")
    print(f"    section        : {metadata.get('section', '')}")
    print(f"    source_file    : {metadata.get('source_file', '')}")
    print(f"    chapter_gate   : {metadata.get('chapter_gate', '')}")
    print(f"    spoiler_level  : {metadata.get('spoiler_level', '')}")
    print(f"    distance       : {distance}")
    print(f"    tags           : {metadata.get('tags', '')}")
    print(f"    related        : {metadata.get('related_concepts', '')}")
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


def run_query(
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

    result = collection.query(**kwargs)
    return result


def validate_result_shape(result: Dict[str, Any]) -> None:
    for key in ["documents", "metadatas", "distances"]:
        if key not in result:
            raise ValueError(f"查詢結果缺少 `{key}`")

    if not result["documents"] or not result["documents"][0]:
        print("查無結果。")
        raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser(description="Query local Chroma RAG collection.")
    parser.add_argument("query", type=str, help="User query text")
    parser.add_argument("--persist-dir", type=str, default=str(DEFAULT_PERSIST_DIR), help="Chroma persist directory")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME, help="Collection name")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Sentence Transformers model name")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to retrieve")
    parser.add_argument("--doc-type", type=str, choices=["concept", "lore"], help="Filter by document_type")
    parser.add_argument("--chapter-gate", type=int, help="Filter by chapter_gate")
    parser.add_argument("--spoiler-level", type=str, help="Filter by spoiler_level")
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    parser.add_argument("--context-only", action="store_true", help="Print prompt-ready retrieved context only")
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    where = parse_where_filters(args)

    print(f"查詢問題：{args.query}")
    print(f"collection：{args.collection}")
    print(f"model：{args.model}")
    print(f"top_k：{args.top_k}")
    print(f"persist_dir：{persist_dir}")
    if where:
        print(f"where filter：{json.dumps(where, ensure_ascii=False)}")
    print()

    result = run_query(
        persist_dir=persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        query_text=args.query,
        top_k=args.top_k,
        where=where,
    )

    validate_result_shape(result)

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    if args.json:
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