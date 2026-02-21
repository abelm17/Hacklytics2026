"""Headless pipeline runner — no UI required."""
import argparse
import time
import pandas as pd
from pipeline.ingest import load_images
from pipeline.features import extract_all_features
from pipeline.embeddings import compute_clip_embeddings
from pipeline.cluster import cluster_images
from pipeline.model import train_ranker, predict_scores, select_images, precision_at_k
from pipeline.explainer import generate_shap_plots
from pipeline.output import save_results, copy_to_folders


def run(folder: str, copy: bool = False, user_selected: list[str] = None):
    t0 = time.time()

    print("📂 Loading images...")
    records = load_images(folder)
    if not records:
        print("No images found. Exiting.")
        return

    print("🔬 Extracting features...")
    extract_all_features(records)

    print("🧠 Computing CLIP embeddings...")
    embeddings = compute_clip_embeddings(records)

    print("🗂️ Clustering...")
    cluster_images(records, embeddings)

    df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("array", "embedding")}
        for r in records
    ])

    print("🤖 Training ranker...")
    model, scaler, feat_cols = train_ranker(df, user_selected)
    df["predicted_score"] = predict_scores(model, scaler, df)
    df = select_images(df)

    print("📊 Generating SHAP plots...")
    generate_shap_plots(model, scaler, df, feat_cols)

    print("💾 Saving results...")
    save_results(df)
    if copy:
        copy_to_folders(df)

    elapsed = time.time() - t0
    p_at_3 = precision_at_k(df, k=3)
    print(f"\n{'='*40}")
    print(f"✅ Done in {elapsed:.1f}s")
    print(f"   Total images : {len(df)}")
    print(f"   Selected     : {df['selected'].sum()}")
    print(f"   Clusters     : {df['cluster'].nunique()}")
    print(f"   Precision@3  : {p_at_3:.3f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo Ranker — headless mode")
    parser.add_argument("--folder", required=True, help="Path to image folder")
    parser.add_argument("--copy", action="store_true", help="Copy images to Selected/Rejected")
    args = parser.parse_args()
    run(args.folder, copy=args.copy)
