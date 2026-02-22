PicMyPic

AI-powered photo selection pipeline for Georgia Tech's Hacklytics. Processes 2–3000 raw or JPEG images, extracts visual features, clusters similar shots, and ranks them using an XGBoost model with SHAP explainability.

-----------

## How to get started

git clone https://github.com/abelm17/Hacklytics2026

cd Hacklytics2026

### 1. Install dependencies
*** MUST USE Python 3.10.11 ***

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pre-download CLIP model (optional but recommended)
```bash
python -c "
from transformers import CLIPModel, CLIPProcessor
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
"
```

### 3. Launch the UI
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

------------

## Our Project Structure

```
photo_ranker/
├── app.py                    # Streamlit UI
├── run_pipeline.py           # Headless CLI runner
├── config.py                 # Tunable constants
├── requirements.txt
├── README.md
├── pipeline/
│   ├── ingest.py             # Image loading & resizing
│   ├── features.py           # CV feature extraction + face analysis
│   ├── embeddings.py         # CLIP embeddings (batched)
│   ├── cluster.py            # pHash + DBSCAN clustering
│   ├── model.py              # XGBoost ranker + pseudo-labels
│   ├── explainer.py          # SHAP plots
│   └── output.py             # CSV/JSON/folder output
├── outputs/                  # Auto-generated results
│   ├── results.csv
│   ├── results.json
│   ├── shap_summary.png
│   ├── shap_importance.png
│   ├── Selected/
│   └── Rejected/
└── sample_images/            # Put your test JPEGs here
```

-------------

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `MAX_WIDTH` | 1500 | Max image width for processing |
| `BATCH_SIZE` | 32 | CLIP embedding batch size |
| `CLUSTER_EPS` | 0.35 | DBSCAN epsilon (cosine distance) |
| `PHASH_THRESHOLD` | 10 | Hamming distance for near-duplicate detection |
| `TOP_K_PER_CLUSTER` | 1 | Best images selected per cluster |
| `SCORE_THRESHOLD` | 0.5 | Min score to be auto-selected |
| `CLIP_MODEL` | `clip-vit-base-patch32` | Hugging Face model ID |
| `FACE_CONFIDENCE` | 0.7 | Threshold for facial confidence |
| `RANDOM_SEED` | 42 | Random Seed |


-----------

## Extracted Feature

| Feature | Method |
|---|---|
| Sharpness | Laplacian variance |
| Brightness | Mean pixel intensity |
| Brightness variance | Pixel variance |
| Contrast | Pixel std deviation |
| Color entropy | Per-channel histogram entropy |
| Face count | MediaPipe FaceDetection |
| Largest face area ratio | Bounding box area |
| Eye openness | MediaPipe FaceMesh EAR |

-------------

## Performance

| Images | CPU time (approx) |
|---|---|
| 500 | ~75 seconds |
| 1000 | ~2.5 minutes |
| 3000 | ~7 minutes |

GPU (CUDA) reduces CLIP embedding time by ~4–6x.

-----------

## Output Files

- `outputs/results.csv` — all images with scores, clusters, selected flag
- `outputs/results.json` — same data as JSON
- `outputs/shap_importance.png` — feature importance bar chart
- `outputs/shap_summary.png` — SHAP beeswarm summary
- `outputs/Selected/` — top-ranked images (if `--copy` used)
- `outputs/Rejected/` — remaining images (if `--copy` used)
