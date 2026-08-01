import os
import sys
import time
import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# 不再用 chromadb / SentenceTransformerEmbeddingFunction
from simple_vector_store import load_index, search

from generate_answer import (
    DEFAULT_EMBED_MODEL_NAME,      # 其實現在用不到，但保留沒關係
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_BASE_URL,
    build_context_blocks,
    format_context_for_prompt,
    build_system_prompt,
    build_user_prompt,
    build_citations,
    call_gemini_chat,
    normalize_answer_payload,
    citation_lookup,
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI()

_gemini_client = None  # 目前 call_gemini_chat 自己 new client，用不到，但保留結構


@app.on_event("startup")
def load_resources():
    global _gemini_client

    print("[STARTUP] 載入向量索引與 embedding 模型...", file=sys.stderr)
    t0 = time.time()

    # simple_vector_store：在這裡預載索引與模型，只跑一次
    load_index()

    _gemini_client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=DEFAULT_GEMINI_BASE_URL,
    )

    t1 = time.time()
    print(f"[STARTUP] 載入完成，耗時 {t1 - t0:.2f}s", file=sys.stderr)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


@app.post("/ask")
def ask(req: QueryRequest):
    try:
        print(f"[DEBUG] 收到請求: {req.query}", file=sys.stderr)

        # ---- 向量檢索：改用 simple_vector_store.search ----
        t0 = time.time()
        print("[DEBUG] 開始向量檢索...", file=sys.stderr)
        documents, metadatas, distances = search(
            query=req.query,
            top_k=req.top_k,
        )
        t1 = time.time()
        print(f"[DEBUG] 向量檢索完成，耗時 {t1 - t0:.2f}s", file=sys.stderr)
        print(f"[DEBUG] documents 數量: {len(documents)}", file=sys.stderr)

        if not documents:
            print("[DEBUG] documents 為空，回傳資料不足", file=sys.stderr)
            return {
                "question": req.query,
                "short_answer": "目前資料不足以回答這個問題。",
                "detailed_answer": "目前資料不足以回答這個問題。",
                "matched_citations": [],
                "timing": {"retrieval": round(t1 - t0, 3)},
            }

        # ---- Prompt 組裝 ----
        t2 = time.time()
        print("[DEBUG] 開始組合 Prompt...", file=sys.stderr)
        context_blocks = build_context_blocks(documents, metadatas, distances)
        context_text = format_context_for_prompt(context_blocks)
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(req.query, context_text)
        citations = build_citations(context_blocks)
        t3 = time.time()
        print(f"[DEBUG] Prompt 組裝完成，耗時 {t3 - t2:.2f}s", file=sys.stderr)

        # ---- 呼叫 Gemini ----
        t4 = time.time()
        print("[DEBUG] 開始呼叫 Gemini...", file=sys.stderr)
        answer_text = call_gemini_chat(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=DEFAULT_GEMINI_BASE_URL,
            model=DEFAULT_GEMINI_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=700,
        )
        t5 = time.time()
        print(f"[DEBUG] Gemini 回應完成，耗時 {t5 - t4:.2f}s", file=sys.stderr)

        # ---- 後處理 ----
        print("[DEBUG] 開始後處理...", file=sys.stderr)
        structured = normalize_answer_payload(answer_text)
        lookup = citation_lookup(citations)
        matched_citations = [
            lookup[label] for label in structured["citation_labels"] if label in lookup
        ]
        print("[DEBUG] 後處理完成，準備回傳結果", file=sys.stderr)

        return {
            "question": req.query,
            "short_answer": structured["short_answer"],
            "detailed_answer": structured["detailed_answer"],
            "matched_citations": matched_citations,
            "timing": {
                "retrieval": round(t1 - t0, 3),
                "prompt_build": round(t3 - t2, 3),
                "gemini": round(t5 - t4, 3),
                "total": round(t5 - t0, 3),
            },
        }

    except Exception as e:
        import traceback
        print("[ERROR] /ask 發生例外，完整 Traceback 如下：", file=sys.stderr)
        traceback.print_exc()
        return {"error": str(e), "type": type(e).__name__}