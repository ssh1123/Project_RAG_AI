#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# 改成使用 simple_vector_store
from simple_vector_store import load_index, search

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TOP_K = 4


def build_context_blocks(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    distances: List[Any],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

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
    parts: List[str] = []

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
            "citations": ["[來源1]", "[來源2]"],
        },
        "citations": citations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RAG answer skeleton from simple_vector_store retrieval."
    )
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Print prompt-ready context only",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print system prompt + user prompt only",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    # 啟動時載入 simple_index.npz 與 embedding 模型
    load_index()

    documents, metadatas, distances = search(
        query=args.query,
        top_k=args.top_k,
    )

    if not documents:
        output = {
            "question": args.query,
            "error": "查無相關 context",
            "system_prompt": build_system_prompt(),
            "user_prompt": "",
            "retrieved_context": [],
            "citations": [],
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