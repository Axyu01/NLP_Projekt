import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PKL_PATH = '/Users/nataliamaciewicz/Documents/studia/SM/SEM 2/NLP/NLP_Projekt/test_data/reconstruction/reconstruction.pkl'

OUT_DIR = "."
SUMMARY_CSV = os.path.join(OUT_DIR, "pytanie1_summary.csv")
TRIALS_CSV  = os.path.join(OUT_DIR, "pytanie1_trials.csv")
SUMMARY_TXT = os.path.join(OUT_DIR, "pytanie1_summary.txt")

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

original_text = data["original_text"]
original_index = data.get("original_index", None)
original_embed = np.array(data["original_embed"]).reshape(1, -1)

rows = []
for i, trial in enumerate(data["reconstruction_trials"], start=1):
    end_text = trial["end_result"]
    end_embed = np.array(trial["end_embed"]).reshape(1, -1)
    sim = float(cosine_similarity(original_embed, end_embed)[0, 0])

    rows.append({
        "trial_id": i,
        "cosine": sim,
        "reconstructed_text": end_text
    })

df_trials = pd.DataFrame(rows).sort_values("cosine", ascending=False).reset_index(drop=True)

best = df_trials.iloc[0]
worst = df_trials.iloc[-1]

df_summary = pd.DataFrame([{
    "original_index": original_index,
    "original_text": original_text,
    "best_text": best["reconstructed_text"],
    "best_cosine": best["cosine"],
    "worst_text": worst["reconstructed_text"],
    "worst_cosine": worst["cosine"],
    "mean_cosine": float(df_trials["cosine"].mean()),
    "std_cosine": float(df_trials["cosine"].std(ddof=0))
}])

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)

print("\n=== PYTANIE 1: TABELA GŁÓWNA (1 WIERSZ) ===\n")
print(df_summary.to_string(index=False))

print("\n=== PYTANIE 1: TABELA 10 PRÓB (posortowana po cosine) ===\n")
print(df_trials[["trial_id", "cosine"]].to_string(index=False))

df_summary.to_csv(SUMMARY_CSV, index=False)
df_trials.to_csv(TRIALS_CSV, index=False)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("PYTANIE 1 — PODSUMOWANIE REKONSTRUKCJI\n\n")
    f.write(f"Original index: {original_index}\n\n")
    f.write("ORIGINAL TEXT:\n")
    f.write(original_text + "\n\n")

    f.write(f"BEST COSINE: {best['cosine']:.6f}\n")
    f.write("BEST TEXT:\n")
    f.write(best["reconstructed_text"] + "\n\n")

    f.write(f"WORST COSINE: {worst['cosine']:.6f}\n")
    f.write("WORST TEXT:\n")
    f.write(worst["reconstructed_text"] + "\n\n")

    f.write(f"MEAN COSINE: {df_summary.loc[0,'mean_cosine']:.6f}\n")
    f.write(f"STD  COSINE: {df_summary.loc[0,'std_cosine']:.6f}\n")

print(f"\nZapisano pliki:")
print(f"- {SUMMARY_CSV}")
print(f"- {TRIALS_CSV}")
print(f"- {SUMMARY_TXT}")
