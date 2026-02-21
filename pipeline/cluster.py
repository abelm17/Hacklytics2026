import numpy as np
import imagehash
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from config import CLUSTER_EPS, CLUSTER_MIN_SAMPLES, PHASH_THRESHOLD

def compute_phashes(records: list[dict]) -> list:
    return [imagehash.phash(Image.fromarray(r["array"])) for r in records]

def cluster_images(records: list[dict], embeddings: np.ndarray) -> list[int]:
    """
    Two-pass clustering:
    1. Near-duplicate detection via pHash
    2. DBSCAN on CLIP embeddings for semantic grouping
    """
    n = len(records)

    hashes = compute_phashes(records)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= PHASH_THRESHOLD:
                union(i, j)

    root_map = {}
    dup_labels = []
    for i in range(n):
        root = find(i)
        if root not in root_map:
            root_map[root] = len(root_map)
        dup_labels.append(root_map[root])

    emb_norm = normalize(embeddings)
    db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES, metric="cosine", n_jobs=-1)
    semantic_labels = db.fit_predict(emb_norm)

    max_dup = max(dup_labels) + 1
    final_labels = []
    for i in range(n):
        siblings = [j for j in range(n) if dup_labels[j] == dup_labels[i]]
        if len(siblings) > 1:
            final_labels.append(dup_labels[i])
        else:
            sem = semantic_labels[i]
            final_labels.append(max_dup + sem if sem >= 0 else max_dup + 9999 + i)

    unique = {}
    result = []
    for lbl in final_labels:
        if lbl not in unique:
            unique[lbl] = len(unique)
        result.append(unique[lbl])

    for i, rec in enumerate(records):
        rec["cluster"] = result[i]

    return result
