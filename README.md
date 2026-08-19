# RAG_AI

RAG_AI 是一個用來支援 **Wind Village** 專案的 RAG（Retrieval-Augmented Generation）後端工具集合。它負責建立知識庫、產生向量嵌入、查詢 RAG 結果，並用 Gemini 產生回答。

## 專案結構

```text
RAG_AI/
├── scripts/
│   ├── generate_answer.py
│   ├── build_chunks.py
│   ├── build_embeddings.py
│   ├── query_rag.py
│   └── answer_rag.py
│   └──...
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

## data 欄位說明

`data/` 底下的資料主要用來建立知識庫，通常會整理成 JSON 檔。為了方便隊友擴充資料集，建議每份資料都盡量遵守相同格式。

### 常見欄位說明

以下欄位是目前資料集中常見、或建議保留的欄位：

- `id`：資料的唯一識別碼。
- `type`：資料類型，例如 `concept`、`lore`、`npc`、`quest`。
- `title`：標題。
- `summary`：簡短摘要，適合快速理解這筆資料的重點。
- `definition`：概念定義或詳細說明。
- `importance`：為什麼這筆資料重要。
- `town_example`：在風待村中的案例說明。
- `misconception`：常見誤解或注意事項。
- `related_concepts`：相關概念清單。
- `keywords`：檢索關鍵字。
- `suggested_questions`：可直接拿來測試 RAG 的問題。
- `location_id`：關聯地點 ID。
- `npc_id`：關聯 NPC ID。
- `quest_id`：關聯任務 ID。
- `chapter_gate`：章節限制，數字越大代表越晚期才解鎖。
- `spoiler_level`：劇情劇透等級，例如 `safe`、`minor`、`spoiler`。
- `difficulty`：閱讀難度，例如 `beginner`、`intermediate`、`advanced`。
- `tags`：標籤清單。
- `content`：可直接作為 chunk 內容的長文。

### 建議欄位原則

- `id` 必填，而且必須唯一。
- `type` 建議必填，方便後續分類與過濾。
- `title`、`summary`、`content` 建議至少保留一個完整描述欄位。
- `keywords` 與 `tags` 建議盡量補齊，會直接影響檢索效果。
- 若是有劇情限制的資料，請務必加入 `chapter_gate` 和 `spoiler_level`。

## data 模板

以下提供幾種常見資料模板，隊友可以直接複製後新增內容。

### 1. 概念資料模板

適合像「地方創生」、「關係人口」、「地方品牌」這類知識型資料。

```json
{
  "id": "concept_example",
  "type": "concept",
  "title": "概念名稱",
  "summary": "一句話摘要。",
  "definition": "這個概念的正式定義。",
  "importance": "這個概念為什麼重要。",
  "town_example": "在風待村中的案例或情境。",
  "misconception": "常見誤解或需要注意的地方。",
  "related_concepts": [
    "相關概念 A",
    "相關概念 B"
  ],
  "keywords": [
    "關鍵字 1",
    "關鍵字 2",
    "關鍵字 3"
  ],
  "suggested_questions": [
    "可以直接拿來測試的問題 1",
    "可以直接拿來測試的問題 2"
  ],
  "location_id": [],
  "npc_id": [],
  "quest_id": [],
  "chapter_gate": 1,
  "spoiler_level": "safe",
  "difficulty": "beginner",
  "tags": [
    "concept",
    "標籤 A",
    "標籤 B"
  ],
  "content": "可直接用來建立 chunk 的完整說明文字。"
}
```

### 2. 劇情 / 世界觀資料模板

適合村落背景、事件經過、地點介紹等內容。

```json
{
  "id": "lore_example",
  "type": "lore",
  "title": "資料標題",
  "summary": "這段劇情或世界觀的簡短摘要。",
  "definition": "補充說明（可選）。",
  "importance": "這段內容在世界觀中的意義。",
  "town_example": "在風待村中的對應情境。",
  "misconception": "常見誤解（可選）。",
  "related_concepts": [
    "相關概念 A",
    "相關概念 B"
  ],
  "keywords": [
    "關鍵字 1",
    "關鍵字 2"
  ],
  "suggested_questions": [
    "可以測試的問題 1",
    "可以測試的問題 2"
  ],
  "location_id": ["location_001"],
  "npc_id": ["npc_001"],
  "quest_id": ["quest_001"],
  "chapter_gate": 2,
  "spoiler_level": "safe",
  "difficulty": "beginner",
  "tags": [
    "lore",
    "世界觀"
  ],
  "content": "完整故事內容或世界觀描述。"
}
```

### 3. NPC 資料模板

適合人物角色設定。

```json
{
  "id": "npc_example",
  "type": "npc",
  "title": "角色名稱",
  "summary": "角色一句話簡介。",
  "definition": "角色背景補充。",
  "importance": "這個角色在劇情或系統中的作用。",
  "town_example": "角色在村子中的活動情境。",
  "misconception": "常見誤解（可選）。",
  "related_concepts": [
    "相關概念 A"
  ],
  "keywords": [
    "角色名",
    "身份",
    "職業"
  ],
  "suggested_questions": [
    "這個角色是誰？",
    "這個角色在做什麼？"
  ],
  "location_id": ["location_001"],
  "npc_id": [],
  "quest_id": [],
  "chapter_gate": 1,
  "spoiler_level": "safe",
  "difficulty": "beginner",
  "tags": [
    "npc",
    "角色"
  ],
  "content": "角色的完整設定與描述。"
}
```

### 4. 任務資料模板

適合支線任務、主線任務、教學任務等。

```json
{
  "id": "quest_example",
  "type": "quest",
  "title": "任務名稱",
  "summary": "任務簡介。",
  "definition": "任務背景說明。",
  "importance": "這個任務為什麼重要。",
  "town_example": "任務在風待村中的實際情境。",
  "misconception": "常見誤解（可選）。",
  "related_concepts": [
    "相關概念 A"
  ],
  "keywords": [
    "任務名",
    "行動內容",
    "目標"
  ],
  "suggested_questions": [
    "這個任務在做什麼？",
    "這個任務要解決什麼問題？"
  ],
  "location_id": ["location_001"],
  "npc_id": ["npc_001"],
  "quest_id": [],
  "chapter_gate": 3,
  "spoiler_level": "minor",
  "difficulty": "intermediate",
  "tags": [
    "quest",
    "任務"
  ],
  "content": "任務完整說明與流程內容。"
}
```

## 新增資料時的建議

### 建議寫法

- 每筆資料盡量只講一個主題。
- `title` 和 `summary` 要短而清楚。
- `content` 裡可以寫完整內容，但要保持條理。
- `keywords` 與 `suggested_questions` 越完整，RAG 檢索通常越好。
- 若資料涉及劇情，請務必標示 `chapter_gate` 與 `spoiler_level`。

### 不建議寫法

- 不要把太多不同主題塞進同一筆資料。
- 不要省略 `id`。
- 不要讓 `type` 混亂，否則後續篩選會很難維護。
- 不要把 `content` 寫得太短，否則切 chunk 後資訊會不足。

## 使用方式

### 1. 建立虛擬環境

```powershell
python -m venv env_rag
```

### 2. 安裝套件

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
.\env_rag\Scripts\python.exe scripts\build_index.py
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
- (已改成使用simple_vector_store，因chroma不相容)
- `.env` 不要提交到 Git，避免洩漏 API Key。
- 如果要讓 Unity 只顯示簡短回答，可以先只讀 `short_answer` 欄位。
- 若隊友要新增資料，請盡量依照上面的模板格式建立，避免資料結構不一致。

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
3. 執行 `build_index.py`。
4. 用 `query_rag.py` 確認檢索結果正確。
5. 用 `answer_rag.py` 或 `generate_answer.py` 產生答案。
6. 最後再接 Unity 前端。

這樣比較容易除錯，也方便之後維護。
