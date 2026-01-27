import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


TRAIN_NPZ = "../../test_data/oversampling/training_data.npz"
LLM_NPZ   = "../../test_data/oversampling/os_LLM.npz"

N_EXAMPLES = 10

MAX_CHARS = 140

OUT_MAIN_CSV = "pytanie2_llm_examples.csv"
OUT_MAIN_TXT = "pytanie2_llm_examples_full.txt"
OUT_SUMMARY_CSV = "pytanie2_llm_similarity_summary.csv"


def shorten(text: str, max_chars: int = 140) -> str:
    if text is None:
        return ""
    t = str(text).replace("\n", " ").strip()
    return t if len(t) <= max_chars else t[:max_chars] + "…"

def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0, 0])


train = np.load(TRAIN_NPZ, allow_pickle=True)
X_train = train["X"].tolist()
id_train = train["id"].tolist()
emb_train = train["embeds"]

y_train = train["y"]

centroid_NOT = np.mean(emb_train[np.array(y_train) == 0], axis=0)
centroid_OFF = np.mean(emb_train[np.array(y_train) == 1], axis=0)


id_to_text = {int(i): str(t) for i, t in zip(id_train, X_train)}
id_to_emb  = {int(i): emb for i, emb in zip(id_train, emb_train)}

print(f"Loaded training_data.npz: {len(id_to_text)} rows")

llm = np.load(LLM_NPZ, allow_pickle=True)
samples = llm["samples"].tolist()
synth_embeds = llm["embeds"]
p1_ids = llm["parent1_id"].tolist()
p2_ids = llm["parent2_id"].tolist()

print(f"Loaded os_LLM.npz: {len(samples)} synthetic samples")

rows = []

missing_parents = 0

for gen_text, gen_emb, p1, p2 in zip(samples, synth_embeds, p1_ids, p2_ids):
    p1 = int(p1)
    p2 = int(p2)

    if p1 not in id_to_text or p2 not in id_to_text or p1 not in id_to_emb or p2 not in id_to_emb:
        missing_parents += 1
        continue

    parent1_text = id_to_text[p1]
    parent2_text = id_to_text[p2]
    parent1_emb  = id_to_emb[p1]
    parent2_emb  = id_to_emb[p2]

    sim1 = cos(gen_emb, parent1_emb)
    sim2 = cos(gen_emb, parent2_emb)
    sim_avg = (sim1 + sim2) / 2.0

    sim_cent_off = cos(gen_emb, centroid_OFF)
    sim_cent_not = cos(gen_emb, centroid_NOT)

    pred_class = "OFF" if sim_cent_off > sim_cent_not else "NOT"
    margin = sim_cent_off - sim_cent_not


    rows.append({
        "parent1_id": p1,
        "parent1_text": parent1_text,
        "parent2_id": p2,
        "parent2_text": parent2_text,
        "generated_text": str(gen_text),
        "sim_to_p1": sim1,
        "sim_to_p2": sim2,
        "sim_avg": sim_avg,

        "sim_to_centroid_OFF": sim_cent_off,
        "sim_to_centroid_NOT": sim_cent_not,
        "pred_class": pred_class,
        "margin": margin
    })

df_all = pd.DataFrame(rows)

print(f"Built full table: {len(df_all)} rows (skipped missing parents: {missing_parents})")

if len(df_all) == 0:
    raise RuntimeError("No rows created. Check if parent IDs match training_data.npz IDs.")


df_sorted = df_all.sort_values("sim_avg", ascending=False).reset_index(drop=True)

k = min(N_EXAMPLES, len(df_sorted))
if k < 5:
    df_pick = df_sorted.head(k).copy()
else:
    top_n = k // 3
    bot_n = k // 3
    mid_n = k - top_n - bot_n

    top = df_sorted.head(top_n)
    bottom = df_sorted.tail(bot_n)

    mid_start = (len(df_sorted) - mid_n) // 2
    mid = df_sorted.iloc[mid_start: mid_start + mid_n]

    df_pick = pd.concat([top, mid, bottom], axis=0).drop_duplicates().reset_index(drop=True)

df_report = df_pick.copy()
df_report["parent1_text"] = df_report["parent1_text"].apply(lambda t: shorten(t, MAX_CHARS))
df_report["parent2_text"] = df_report["parent2_text"].apply(lambda t: shorten(t, MAX_CHARS))
df_report["generated_text"] = df_report["generated_text"].apply(lambda t: shorten(t, MAX_CHARS))

df_report = df_report[
    ["parent1_id", "parent1_text",
     "parent2_id", "parent2_text",
     "generated_text",
     "sim_to_p1", "sim_to_p2", "sim_avg",
     "sim_to_centroid_OFF", "sim_to_centroid_NOT",
     "pred_class", "margin"]
].sort_values("sim_avg", ascending=False)


df_report.to_csv(OUT_MAIN_CSV, index=False)
print(f"Saved main examples table: {OUT_MAIN_CSV}")

summary = pd.DataFrame([{
    "count": int(len(df_all)),
    "sim_to_p1_mean": float(df_all["sim_to_p1"].mean()),
    "sim_to_p1_std":  float(df_all["sim_to_p1"].std(ddof=0)),
    "sim_to_p1_min":  float(df_all["sim_to_p1"].min()),
    "sim_to_p1_max":  float(df_all["sim_to_p1"].max()),
    "sim_to_p2_mean": float(df_all["sim_to_p2"].mean()),
    "sim_to_p2_std":  float(df_all["sim_to_p2"].std(ddof=0)),
    "sim_to_p2_min":  float(df_all["sim_to_p2"].min()),
    "sim_to_p2_max":  float(df_all["sim_to_p2"].max()),
    "sim_avg_mean":   float(df_all["sim_avg"].mean()),
    "sim_avg_std":    float(df_all["sim_avg"].std(ddof=0)),
    "sim_avg_min":    float(df_all["sim_avg"].min()),
    "sim_avg_max":    float(df_all["sim_avg"].max()),
}])

summary.to_csv(OUT_SUMMARY_CSV, index=False)
print(f"Saved similarity summary table: {OUT_SUMMARY_CSV}")

with open(OUT_MAIN_TXT, "w", encoding="utf-8") as f:
    f.write("PYTANIE 2 — PRZYKŁADY OVERSAMPLINGU LLM (PEŁNE TEKSTY)\n\n")
    for idx, row in df_pick.sort_values("sim_avg", ascending=False).iterrows():
        f.write(f"--- EXAMPLE ---\n")
        f.write(f"parent1_id: {row['parent1_id']} | sim_to_p1: {row['sim_to_p1']:.6f}\n")
        f.write(f"parent2_id: {row['parent2_id']} | sim_to_p2: {row['sim_to_p2']:.6f}\n")
        f.write(f"sim_avg: {row['sim_avg']:.6f}\n\n")
        f.write("PARENT 1 TEXT:\n")
        f.write(row["parent1_text"] + "\n\n")
        f.write("PARENT 2 TEXT:\n")
        f.write(row["parent2_text"] + "\n\n")
        f.write("GENERATED TEXT:\n")
        f.write(row["generated_text"] + "\n\n")

print(f"Saved full-text appendix: {OUT_MAIN_TXT}")

pd.set_option("display.max_colwidth", 80)
print("\n=== MAIN TABLE (report version, shortened texts) ===")
print(df_report.to_string(index=False))

print("\n=== SUMMARY TABLE ===")
print(summary.to_string(index=False))
