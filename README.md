# RAG_AI

RAG_AI 是一個用來支援 **Wind Village** 專案的 RAG（Retrieval-Augmented Generation）後端工具集合。它負責建立知識庫、產生向量嵌入、查詢 RAG 結果，並用 Gemini 產生回答。123

## 專案結構

```text
RAG_AI/
├── scripts/
│   ├── generate_answer.py
│   ├── build_chunks.py
│   ├── build_embeddings.py
│   ├── query_rag.py
│   └── answer_rag.py
├── chroma_db/
├── data/
├── .env
├── requirements.txt
└── README.md
```

## 目錄說明

### `scripts/`
這個資料夾放 RAG 流程相關的 Python 腳本：

- `build_chunks.py`：將原始資料切成適合檢索的小段落。
- `build_embeddings.py`：把 chunk 轉成向量並寫入 ChromaDB。
- `query_rag.py`：查詢 ChromaDB 中的相關內容。
- `answer_rag.py`：根據檢索結果產生回答。
- `generate_answer.py`：整合檢索與生成的主要執行腳本。

### `chroma_db/`
存放 ChromaDB 向量資料庫檔案。

### `data/`
存放原始資料、概念資料、劇情資料或其他輸入來源。

### `.env`
存放環境變數，例如 Gemini API Key。

### `requirements.txt`
列出此專案需要安裝的 Python 套件。

## 使用方式

### 1. 建立虛擬環境

```powershell
python -m venv env_rag
```

### 2. 安裝依賴

```powershell
.\env_rag\Scripts\python.exe -m pip install --upgrade pip
.\env_rag\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. 設定 `.env`

在專案根目錄建立 `.env`，並加入 API Key：

```env
GEMINI_API_KEY=你的_api_key
```

### 4. 先建立資料切片

```powershell
.\env_rag\Scripts\python.exe scripts\build_chunks.py
```

### 5. 建立向量嵌入

```powershell
.\env_rag\Scripts\python.exe scripts\build_embeddings.py
```

### 6. 測試查詢

```powershell
.\env_rag\Scripts\python.exe scripts\query_rag.py "什麼是關係人口？"
```

### 7. 產生回答

```powershell
.\env_rag\Scripts\python.exe scripts\answer_rag.py "什麼是關係人口？"
```

### 8. 使用整合版回答腳本

```powershell
.\env_rag\Scripts\python.exe scripts\generate_answer.py "什麼是關係人口？" --pretty
```

## 輸出說明

`generate_answer.py` 的輸出通常會是 JSON 格式，包含像是：

- `question`
- `short_answer`
- `detailed_answer`
- `citation_labels`
- `matched_citations`
- `answer_text`

這樣方便 Unity 或其他前端直接解析。

## 注意事項

- 建議不要把虛擬環境資料夾提交到 Git。
- `chroma_db/` 若是可重建資料，可視專案需求決定是否提交。
- `.env` 不要提交到 Git，避免洩漏 API Key。
- 如果要讓 Unity 只顯示簡短回答，可以先只讀 `short_answer` 欄位。

## 建議的 requirements

如果你還沒有 `requirements.txt`，可以先放這些：

```text
fastapi
uvicorn
chromadb
sentence-transformers
openai
python-dotenv
```

## 除錯建議

如果發生問題，可以依序檢查：

1. `.env` 是否有正確設定 `GEMINI_API_KEY`。
2. `chroma_db/` 是否已成功建立。
3. `build_chunks.py` 與 `build_embeddings.py` 是否執行成功。
4. `generate_answer.py` 是否能單獨正常輸出 JSON。
5. 若是 Unity 串接，確認 Unity 指向的 Python 路徑是否正確。

## 建議流程

一般建議流程是：

1. 先整理 `data/`。
2. 執行 `build_chunks.py`。
3. 執行 `build_embeddings.py`。
4. 用 `query_rag.py` 確認檢索結果正確。
5. 用 `answer_rag.py` 或 `generate_answer.py` 產生答案。
6. 最後再接 Unity 前端。

這樣比較容易除錯，也方便之後維護。
