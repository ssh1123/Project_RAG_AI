import json
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from simple_vector_store import search, load_index  # load_index 可選，用來預先載入

from generate_answer import (
    DEFAULT_TOP_K,
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_API_KEY_ENV,

    build_context_blocks,
    format_context_for_prompt,
    build_system_prompt,
    build_user_prompt,
    build_citations,
    call_gemini_chat,
    try_parse_json_answer,
    fallback_parse_answer,
)

app = FastAPI()

# 全域快取：只在 FastAPI 啟動時載入一次
_embedding_function = None
_collection = None

_index_ready = False

@app.on_event("startup")
def load_model_and_index():
    global _index_ready
    print("正在載入 simple_index 和 embedding 模型...")
    load_index()           # 這行會跑 simple_vector_store.load_index()
    _index_ready = True
    print("載入完成，server 已就緒。")

class AskRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K


class AskResponse(BaseModel):
    question: str
    short_answer: str
    #detailed_answer: str
    timing: dict


@app.get("/")
def root():
    return {"message": "AI RAG server is running"}


@app.get("/health")
def health():
    return {"status": "ok" if _index_ready else "loading"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        if not _index_ready:
            raise RuntimeError("索引尚未載入完成，請稍後再試。")

        t0 = time.time()

        documents, metadatas, distances = search(
            query=req.question,
            top_k=req.top_k,
        )
        t1 = time.time()

        context_blocks = build_context_blocks(documents, metadatas, distances)
        context_text = format_context_for_prompt(context_blocks)

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(req.question, context_text)

        t2 = time.time()
        raw_answer = call_gemini_chat(
            api_key=DEFAULT_API_KEY_ENV,
            base_url=DEFAULT_GEMINI_BASE_URL,
            model=DEFAULT_GEMINI_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=800,
        )
        t3 = time.time()

        data = try_parse_json_answer(raw_answer) or fallback_parse_answer(raw_answer)

        return {
            "question": req.question,
            "short_answer": data.get("short_answer", ""),
            #"detailed_answer": data.get("detailed_answer", ""),
            "timing": {
                "retrieval": round(t1 - t0, 3),
                "prompt_build": round(t2 - t1, 3),
                "gemini": round(t3 - t2, 3),
                "total": round(t3 - t0, 3),
            },
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))