import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import tempfile
import shutil
from PIL import Image

st.set_page_config(
    page_title="📸 Photo Ranker",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #13131f; }
    [data-testid="stFileUploader"] {
        border: 2px dashed #cba6f7;
        border-radius: 16px;
        padding: 24px;
        background: #1e1e2e;
    }
    [data-testid="stFileUploader"]:hover { border-color: #89b4fa; }
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] { color: #cba6f7; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #a6adc8; }
    h1 { color: #cdd6f4 !important; }
    h2, h3 { color: #bac2de !important; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #cba6f7 0%, #89b4fa 100%);
        color: #1e1e2e;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.6rem;
        font-size: 1rem;
        width: 100%;
    }
    .info-box {
        background: #1e1e2e;
        border-left: 4px solid #89b4fa;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 16px;
        color: #a6adc8;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    .stApp * {
        color: #cdd6f4 !important;   /* light gray */
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📸 Photo Ranker")
st.caption("Upload your photos · AI ranks and selects the best ones · See exactly why")
st.divider()

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    <div class="info-box">
    <b>1. Upload</b> — Select your JPEG photos using the uploader below. Use <b>Ctrl+A / Cmd+A</b> in the file picker to grab a whole folder at once.<br><br>
    <b>2. Analyse</b> — Every photo is scored for sharpness, exposure, face quality, and semantic content via CLIP AI.<br><br>
    <b>3. Cluster</b> — Similar / near-duplicate burst shots are grouped so the best frame from each burst is selected.<br><br>
    <b>4. Rank</b> — XGBoost predicts a quality score. The best from each cluster is selected.<br><br>
    <b>5. Explain</b> — SHAP charts show exactly which features drove each photo's score.<br><br>
    <b>6. Download</b> — Export your selected photos as a zip, or download the full scores CSV.
    </div>
    <b>Supported:</b> JPEG / JPG &nbsp;|&nbsp; <b>Speed:</b> ~1000 photos in 2–3 min on CPU
    """, unsafe_allow_html=True)

# ── Upload zone ───────────────────────────────────────────────────────────────
st.subheader("📂 Step 1 — Upload Your Photos")
st.markdown("""
<div class="info-box">
Click <b>Browse files</b> → go to your photo folder → hit <b>Ctrl+A</b> (Windows) or <b>Cmd+A</b> (Mac) to select everything → click Open.<br>
Or just drag and drop files directly onto the box. Both work.
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    label="Drop photos here, or click Browse files to select your photo folder",
    type=["jpg", "jpeg"],
    accept_multiple_files=True,
    help="Select all JPEGs you want ranked. Ctrl+A or Cmd+A in the file picker selects them all at once.",
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} photo{'s' if len(uploaded_files) != 1 else ''} loaded and ready")

st.divider()

# ── Options ───────────────────────────────────────────────────────────────────
st.subheader("⚙️ Step 2 — Options")
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    top_k = st.slider("Photos to show in results grid", 4, 40, 12, 4)
with col_opt2:
    make_zip = st.checkbox("Create downloadable zip of selected photos", value=True)

with st.expander("🎛 Personalisation — teach it your taste (optional)"):
    st.caption(
        "Run once first. Then paste filenames of photos you liked and re-run to retrain on your preferences."
    )
    user_picks_raw = st.text_area(
        "Filenames you prefer (one per line)",
        height=90,
        placeholder="IMG_0042.jpg\nIMG_0107.jpg",
    )
    user_picks = [x.strip() for x in user_picks_raw.strip().splitlines() if x.strip()]
    if user_picks:
        st.info(f"🎯 {len(user_picks)} preferred photos will personalise the ranking")

st.divider()
run_btn = st.button("🚀 Rank My Photos", type="primary", disabled=not uploaded_files)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn and uploaded_files:
    t_start = time.time()

    # Write uploaded files to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="photo_ranker_")
    for uf in uploaded_files:
        dest = os.path.join(tmp_dir, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())

    with st.status("🔄 Running pipeline…", expanded=True) as status:

        st.write(f"📂 Saved {len(uploaded_files)} photos for processing…")
        from pipeline.ingest import load_images
        records = load_images(tmp_dir)
        if not records:
            st.error("No valid JPEG images found.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()
        st.write(f"✅ Loaded **{len(records)}** images")

        st.write("🔬 Extracting quality features (sharpness, brightness, faces, eye openness)…")
        from pipeline.features import extract_all_features
        extract_all_features(records)
        st.write("✅ Features extracted")

        st.write("🧠 Computing CLIP semantic embeddings…")
        from pipeline.embeddings import compute_clip_embeddings
        embeddings = compute_clip_embeddings(records)
        st.write("✅ Embeddings ready")

        st.write("🗂 Clustering similar / burst shots…")
        from pipeline.cluster import cluster_images
        cluster_images(records, embeddings)
        n_clusters = len(set(r["cluster"] for r in records))
        st.write(f"✅ Found **{n_clusters}** clusters")

        st.write("🏋️ Training XGBoost ranker…")
        df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("array", "embedding")}
            for r in records
        ])
        from pipeline.model import train_ranker, predict_scores, select_images, precision_at_k
        model, scaler, feat_cols = train_ranker(df, user_selected=user_picks or None)
        df["predicted_score"] = predict_scores(model, scaler, df)
        df = select_images(df)
        p3 = precision_at_k(df, k=3)
        st.write(f"✅ Scores computed — Precision@3: **{p3:.3f}**")

        st.write("📊 Generating SHAP explainability charts…")
        os.makedirs("outputs", exist_ok=True)
        from pipeline.explainer import generate_shap_plots, get_top_features_for_image
        shap_values = generate_shap_plots(model, scaler, df, feat_cols)
        st.write("✅ SHAP charts generated")

        st.write("💾 Saving results…")
        from pipeline.output import save_results
        save_results(df)

        # Build selected photos zip
        selected_zip_bytes = None
        if make_zip:
            import zipfile, io
            selected_rows = df[df["selected"]]
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for _, row in selected_rows.iterrows():
                    img_path = os.path.join(tmp_dir, row["filename"])
                    if os.path.exists(img_path):
                        zf.write(img_path, row["filename"])
            selected_zip_bytes = zip_buf.getvalue()
            st.write(f"✅ Zip of {len(selected_rows)} selected photos ready")

        status.update(label="✅ All done!", state="complete")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    elapsed = time.time() - t_start

    st.session_state.update({
        "df": df,
        "shap_values": shap_values,
        "feat_cols": feat_cols,
        "arrays": {r["filename"]: r["array"] for r in records},
        "elapsed": elapsed,
        "selected_zip": selected_zip_bytes,
        "p3": p3,
    })

# ── Results ───────────────────────────────────────────────────────────────────
if "df" in st.session_state:
    df           = st.session_state["df"]
    shap_values  = st.session_state["shap_values"]
    feat_cols    = st.session_state["feat_cols"]
    arrays       = st.session_state["arrays"]
    elapsed      = st.session_state["elapsed"]
    selected_zip = st.session_state.get("selected_zip")

    n_total    = len(df)
    n_sel      = int(df["selected"].sum())
    n_rej      = n_total - n_sel
    n_clusters = df["cluster"].nunique()

    # ── Downloads ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⬇️ Downloads")
    dl1, dl2 = st.columns(2)
    with dl1:
        csv_bytes = df.drop(columns=["array","embedding"], errors="ignore").to_csv(index=False).encode()
        st.download_button(
            "📄 Download Full Results CSV",
            data=csv_bytes,
            file_name="photo_ranker_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        if selected_zip:
            st.download_button(
                f"📦 Download Selected Photos ZIP ({n_sel} photos)",
                data=selected_zip,
                file_name="selected_photos.zip",
                mime="application/zip",
                use_container_width=True,
            )

    # ── Metrics ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Photos", n_total)
    c2.metric("✅ Selected", n_sel)
    c3.metric("❌ Rejected", n_rej)
    c4.metric("📦 Clusters", n_clusters)
    c5.metric("⏱ Time", f"{elapsed:.0f}s")

    # ── Cluster table ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📦 Cluster / Burst Summary")
    st.caption("Similar or near-duplicate shots are grouped. Only the best from each group is selected.")
    cluster_summary = (
        df.groupby("cluster")
        .agg(photos=("filename","count"), best_score=("predicted_score","max"),
             avg_score=("predicted_score","mean"), selected=("selected","sum"))
        .reset_index().sort_values("best_score", ascending=False)
    )
    cluster_summary[["best_score","avg_score"]] = cluster_summary[["best_score","avg_score"]].round(3)
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    # ── SHAP ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Why These Photos? (SHAP Explainability)")
    st.caption("These charts show which image qualities had the most influence on the ranking decisions.")
    ca, cb = st.columns(2)
    if os.path.exists("outputs/shap_importance.png"):
        ca.image("outputs/shap_importance.png", caption="Feature Importance", use_container_width=True)
    if os.path.exists("outputs/shap_summary.png"):
        cb.image("outputs/shap_summary.png", caption="SHAP Summary (purple = high value, red = low)", use_container_width=True)

    # ── Selected grid ─────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"🏆 Top {min(top_k, n_sel)} Selected Photos")
    st.caption("Best photo from each cluster. Green = what boosted the score. Red = what hurt it.")

    top_df = df[df["selected"]].nlargest(top_k, "predicted_score").reset_index(drop=True)
    grid_cols = st.columns(4)

    from pipeline.explainer import get_top_features_for_image
    for i, row in top_df.iterrows():
        with grid_cols[i % 4]:
            try:
                arr = arrays.get(row["filename"])
                if arr is not None:
                    img = Image.fromarray(arr)
                    img.thumbnail((400, 400))
                    st.image(img, use_container_width=True)
            except Exception:
                st.warning(row["filename"])

            score = float(row["predicted_score"])
            st.progress(score, text=f"Score {score:.3f}")
            st.caption(f"Cluster {int(row['cluster'])} · {row['filename']}")

            top_feats = get_top_features_for_image(shap_values, row.name, feat_cols, top_n=3)
            for feat in top_feats:
                icon = "🟢" if feat["shap"] > 0 else "🔴"
                st.caption(f"{icon} {feat['feature']}: {feat['shap']:+.3f}")

    # ── Full table ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 All Photos — Full Scores")
    display_cols = ["filename","cluster","predicted_score","selected",
                    "sharpness","brightness","contrast","face_count","eye_openness"]
    display_df = df[display_cols].sort_values("predicted_score", ascending=False).copy()
    for col in ["predicted_score","brightness","contrast","eye_openness"]:
        display_df[col] = display_df[col].round(3)
    display_df["sharpness"] = display_df["sharpness"].round(1)

    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        column_config={
            "predicted_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.3f"),
            "selected": st.column_config.CheckboxColumn("✅ Selected"),
        },
    )

    # ── Rejected ──────────────────────────────────────────────────────────
    with st.expander(f"🗑 View rejected photos ({n_rej})"):
        rej_df = df[~df["selected"]].nlargest(24, "predicted_score").reset_index(drop=True)
        rej_cols = st.columns(4)
        for i, row in rej_df.iterrows():
            with rej_cols[i % 4]:
                try:
                    arr = arrays.get(row["filename"])
                    if arr is not None:
                        img = Image.fromarray(arr)
                        img.thumbnail((300, 300))
                        st.image(img, use_container_width=True)
                    st.caption(f"Score {row['predicted_score']:.3f} · {row['filename']}")
                except Exception:
                    pass

else:
    # ── Landing state ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
    👆 <b>To get started:</b> click <b>Browse files</b> above, navigate to your photo folder, 
    press <b>Ctrl+A</b> or <b>Cmd+A</b> to select all photos, then click <b>Rank My Photos</b>.<br><br>
    No folders to create. No config needed. Everything is automatic.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown("**🔬 Features extracted**\nSharpness · Brightness · Contrast · Color entropy · Faces · Eye openness")
    col2.markdown("**🤖 AI models used**\nCLIP (semantic understanding) · MediaPipe (face/eye detection) · XGBoost (ranking)")
    col3.markdown("**📊 Outputs**\nSelected photo zip · Full CSV · SHAP charts · Cluster breakdown")
