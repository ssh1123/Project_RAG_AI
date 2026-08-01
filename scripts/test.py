import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 建立簡易向量庫
docs = ["hello world"]
embs = model.encode(docs)  # shape: (N, D)

def search(query, top_k=4):
    q_emb = model.encode([query])[0]  # shape: (D,)
    # 計算 cosine similarity
    sims = embs @ q_emb / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb))
    idx = np.argsort(-sims)[:top_k]
    return [(docs[i], float(sims[i])) for i in idx]

print(search("hello"))