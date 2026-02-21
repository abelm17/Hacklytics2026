import os
import shutil
import json
import pandas as pd

def save_results(df: pd.DataFrame, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    out = df.drop(columns=["array", "embedding"], errors="ignore")
    out.to_csv(f"{output_dir}/results.csv", index=False)
    records = out.to_dict(orient="records")
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Results saved to {output_dir}/")

def copy_to_folders(df: pd.DataFrame, output_dir: str = "outputs"):
    sel_dir = os.path.join(output_dir, "Selected")
    rej_dir = os.path.join(output_dir, "Rejected")
    os.makedirs(sel_dir, exist_ok=True)
    os.makedirs(rej_dir, exist_ok=True)

    for _, row in df.iterrows():
        dest = sel_dir if row["selected"] else rej_dir
        try:
            shutil.copy2(row["path"], os.path.join(dest, row["filename"]))
        except Exception as e:
            print(f"[WARN] Could not copy {row['filename']}: {e}")
