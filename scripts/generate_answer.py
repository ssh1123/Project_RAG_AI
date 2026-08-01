#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

#import chromadb
#from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from simple_vector_store import search
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import sys
import time 

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")




BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = BASE_DIR / "chroma_db"
DEFAULT_COLLECTION_NAME = "kazemachi_game_knowledge"
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_TOP_K = 4
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_API_KEY_ENV = os.getenv("GEMINI_API_KEY")


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
    embed_model_name: str,
    query_text: str,
    top_k: int,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    collection = get_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=embed_model_name,
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
        blocks.append({
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
        })

    return blocks


def format_context_for_prompt(context_blocks: List[Dict[str, Any]]) -> str:
    parts = []

    for block in context_blocks:
        parts.append(
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

    return "\n\n---\n\n".join(parts)


def build_system_prompt() -> str:
    return (
        "你是《風待村》的遊戲內導覽 AI。\n"
        "你的任務是根據提供的檢索內容，回答玩家關於遊戲世界觀、地方創生概念、地點、角色與任務背景的問題。\n\n"
        "回答規則：\n"
        "1. 只能根據提供的 context 回答，不可自行補完未出現在 context 的事實。\n"
        "2. 如果 context 不足以回答，必須明確說「目前資料不足以回答這個問題」。\n"
        "3. 回答要優先使用《風待村》世界觀語氣，但內容仍要清楚、可教學。\n"
        "4. 不可透露未解鎖劇情，不可超出 chapter_gate 限制。\n"
        "5. 不要說你是根據資料庫、檢索結果或 prompt 回答。\n"
        "6. 你必須只輸出合法 JSON，不要輸出 markdown，不要加 ```json。"
    )


def build_user_prompt(user_question: str, context_text: str) -> str:
    return (
        f"玩家問題：{user_question}\n\n"
        f"以下是可用的檢索資料，請只根據這些資料回答：\n\n"
        f"{context_text}\n\n"
        "請只輸出一個 JSON 物件，格式如下：\n"
        "{\n"
        '  "short_answer": "一句到兩句的簡短回答",\n'
        '  "detailed_answer": "較完整的解釋",\n'
        '  "citation_labels": ["來源1", "來源2"]\n'
        "}\n\n"
        "規則：\n"
        "- citation_labels 只能填入提供過的來源標籤，例如 來源1、來源2。\n"
        "- 如果資料不足，short_answer 與 detailed_answer 都要明確表示資料不足。\n"
        "- 不要輸出 JSON 以外的任何文字。"
    )


def build_citations(context_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
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


def call_gemini_chat(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not completion.choices:
        raise RuntimeError(f"Gemini 沒有回傳 choices: {completion}")

    message = completion.choices[0].message
    content = message.content

    if content is None:
        raise RuntimeError(f"Gemini 回傳的 message.content 是 None: {completion}")

    return content.strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def try_parse_json_answer(answer_text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_code_fences(answer_text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None

    return None


def extract_section(text: str, start_label: str, end_labels: List[str]) -> str:
    pattern = rf"{re.escape(start_label)}\s*[:：]\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""

    content = match.group(1).strip()
    earliest_end = None

    for end_label in end_labels:
        end_pattern = rf"\n\s*{re.escape(end_label)}\s*[:：]"
        end_match = re.search(end_pattern, content)
        if end_match:
            pos = end_match.start()
            if earliest_end is None or pos < earliest_end:
                earliest_end = pos

    if earliest_end is not None:
        content = content[:earliest_end].strip()

    return content.strip()


def fallback_parse_answer(answer_text: str) -> Dict[str, Any]:
    short_answer = extract_section(
        answer_text,
        "簡短回答",
        ["詳細說明", "引用來源"]
    )

    detailed_answer = extract_section(
        answer_text,
        "詳細說明",
        ["引用來源"]
    )

    citation_match = re.search(r"引用來源\s*[:：]\s*(.*)", answer_text, re.DOTALL)
    citation_labels: List[str] = []
    if citation_match:
        labels_text = citation_match.group(1)
        citation_labels = re.findall(r"\[?(來源\d+)\]?", labels_text)

    return {
        "short_answer": short_answer or answer_text.strip(),
        "detailed_answer": detailed_answer or "",
        "citation_labels": citation_labels,
    }


def normalize_answer_payload(answer_text: str) -> Dict[str, Any]:
    parsed = try_parse_json_answer(answer_text)
    if parsed is None:
        parsed = fallback_parse_answer(answer_text)

    short_answer = str(parsed.get("short_answer", "")).strip()
    detailed_answer = str(parsed.get("detailed_answer", "")).strip()

    raw_labels = parsed.get("citation_labels", [])
    if isinstance(raw_labels, str):
        raw_labels = re.findall(r"來源\d+", raw_labels)
    elif not isinstance(raw_labels, list):
        raw_labels = []

    citation_labels = []
    for item in raw_labels:
        label = str(item).strip().replace("[", "").replace("]", "")
        if re.fullmatch(r"來源\d+", label):
            citation_labels.append(label)

    citation_labels = list(dict.fromkeys(citation_labels))

    return {
        "short_answer": short_answer,
        "detailed_answer": detailed_answer,
        "citation_labels": citation_labels,
        "answer_text": answer_text.strip(),
    }


def citation_lookup(citations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {c["label"]: c for c in citations}

from simple_vector_store import search

def main():
    parser = argparse.ArgumentParser(description="Generate structured RAG answer with Gemini API.")
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("--persist-dir", type=str, default=str(DEFAULT_PERSIST_DIR), help="Chroma persist directory")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME, help="Collection name")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL_NAME, help="Embedding model name")
    parser.add_argument("--gemini-model", type=str, default=DEFAULT_GEMINI_MODEL, help="Gemini model name")
    parser.add_argument("--api-base", type=str, default=DEFAULT_GEMINI_BASE_URL, help="Gemini OpenAI-compatible base URL")
    parser.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV, help="Environment variable name for Gemini API key")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--doc-type", type=str, choices=["concept", "lore"], help="Filter by document_type")
    parser.add_argument("--chapter-gate", type=int, help="Filter by chapter_gate")
    parser.add_argument("--spoiler-level", type=str, help="Filter by spoiler_level")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=700, help="Max output tokens")
    parser.add_argument("--prompt-only", action="store_true", help="Print prompts without calling Gemini")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()   # <-- 這一行必須存在！

    api_key = os.getenv("GEMINI_API_KEY")
    if not args.prompt_only and not api_key:
        raise ValueError(f"找不到 Gemini API key。請先設定環境變數 `{args.api_key_env}`。")

    persist_dir = Path(args.persist_dir)
    where = parse_where_filters(args)
    t0 = time.time()  # ---- 開始：向量檢索 ----
    # 不再用 query_collection / query_result
    documents, metadatas, distances = search(
        query=args.query,
        top_k=args.top_k,
    )
    t1 = time.time()  # ---- 結束：向量檢索 ----

    # 後面直接用 documents/metadatas/distances
    if not documents:
        output = {
            "question": args.query,
            "short_answer": "目前資料不足以回答這個問題。",
            "detailed_answer": "目前資料不足以回答這個問題。",
            "citation_labels": [],
            "matched_citations": [],
            "answer_text": "目前資料不足以回答這個問題。",
            "citations": [],
            "retrieved_context": [],
            "gemini_model": args.gemini_model,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))
        print(f"[TIMING] 檢索耗時: {t1 - t0:.2f}s (無結果，提早結束)", file=sys.stderr)
        return

    # 後面這幾行保持原樣即可
    t2 = time.time()
    context_blocks = build_context_blocks(documents, metadatas, distances)
    context_text = format_context_for_prompt(context_blocks)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(args.query, context_text)
    citations = build_citations(context_blocks)
    t3 = time.time()
    if args.prompt_only:
        output = {
            "question": args.query,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "citations": citations,
            "retrieved_context": context_blocks,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))
        print(f"[TIMING] 檢索耗時: {t1 - t0:.2f}s | Prompt組裝耗時: {t3 - t2:.2f}s", file=sys.stderr)
        return

    t4 = time.time()  # ---- 開始：Gemini 生成 ----
    answer_text = call_gemini_chat(
        api_key=api_key,
        base_url=args.api_base,
        model=args.gemini_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    t5 = time.time()  # ---- 結束：Gemini 生成 ----

    t6 = time.time()  # ---- 開始：後處理 ----
    structured = normalize_answer_payload(answer_text)
    lookup = citation_lookup(citations)

    matched_citations = [
        lookup[label]
        for label in structured["citation_labels"]
        if label in lookup
    ]
    t7 = time.time()  # ---- 結束：後處理 ----

    output = {
        "question": args.query,
        "short_answer": structured["short_answer"],
        "detailed_answer": structured["detailed_answer"],
        "citation_labels": structured["citation_labels"],
        "matched_citations": matched_citations,
        "answer_text": structured["answer_text"],
        "gemini_model": args.gemini_model,
        "embedding_model": args.embed_model,
        "api_base": args.api_base,
        "citations": citations,
        "retrieved_context": context_blocks,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))

    # ---- 統一輸出各階段耗時 ----
    print(f"[TIMING] 向量檢索: {t1 - t0:.2f}s", file=sys.stderr)
    print(f"[TIMING] Prompt組裝: {t3 - t2:.2f}s", file=sys.stderr)
    print(f"[TIMING] Gemini生成: {t5 - t4:.2f}s", file=sys.stderr)
    print(f"[TIMING] 後處理: {t7 - t6:.2f}s", file=sys.stderr)
    print(f"[TIMING] 總耗時: {t7 - t0:.2f}s", file=sys.stderr)

if __name__ == "__main__":
    main()