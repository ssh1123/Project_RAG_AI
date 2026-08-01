# scripts/build_index.py
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

# 你原本的輸出檔
DATA_FILE = BASE_DIR / "output" / "chunks.jsonl"
INDEX_FILE = BASE_DIR / "data" / "simple_index.npz"


def load_chunks():
    docs = []
    metadatas = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            docs.append(item["text"])
            metadatas.append(item.get("metadata", {}))
    return docs, metadatas


def main():
    print("載入 chunks...")
    docs, metadatas = load_chunks()
    print(f"共 {len(docs)} 個段落")

    print("載入 embedding 模型...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("計算 embeddings...")
    embs = model.encode(docs, convert_to_numpy=True)  # shape: (N, D)

    print("儲存索引到 simple_index.npz")
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        INDEX_FILE,
        embeddings=embs,
        docs=np.array(docs, dtype=object),
        metadatas=np.array(metadatas, dtype=object),
    )
    print("完成")


if __name__ == "__main__":
    main()