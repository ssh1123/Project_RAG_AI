# scripts/simple_vector_store.py
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "data" / "simple_index.npz"

# 啟動時載入索引與模型
_index_loaded = False
_docs = None
_metadatas = None
_embs = None
_model = None

def load_index():
    global _index_loaded, _docs, _metadatas, _embs, _model
    if _index_loaded:
        return

    print("[INDEX] 載入 simple_index.npz...")
    data = np.load(INDEX_FILE, allow_pickle=True)
    _embs = data["embeddings"]          # shape (N, D)
    _docs = data["docs"].tolist()       # list[str]
    _metadatas = data["metadatas"].tolist()  # list[dict]

    print("[INDEX] 載入 embedding 模型...")
    _model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    _index_loaded = True
    print(f"[INDEX] 載入完成，段落數：{len(_docs)}")

def search(query: str, top_k: int = 4):
    """
    回傳 (documents, metadatas, distances) 三個 list，
    結構跟你原本 query_collection 取出的內容一致。
    distances 這裡用 (1 - cosine_similarity) 當成「距離」。
    """
    load_index()

    # 計算 query embedding
    q_emb = _model.encode([query], convert_to_numpy=True)[0]  # shape (D,)
    # 計算 cosine similarity
    emb_norms = np.linalg.norm(_embs, axis=1)
    q_norm = np.linalg.norm(q_emb)
    sims = (_embs @ q_emb) / (emb_norms * q_norm + 1e-8)  # shape (N,)

    # 取前 top_k
    idx = np.argsort(-sims)[:top_k]

    documents = []
    metadatas = []
    distances = []

    for i in idx:
        documents.append(_docs[i])
        metadatas.append(_metadatas[i])
        # 跟原本 Chroma 一樣有個距離值，這裡用 1 - sim
        distances.append(float(1.0 - sims[i]))

    return documents, metadatas, distances