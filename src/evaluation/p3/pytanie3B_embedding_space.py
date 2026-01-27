import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_NPZ = "../../test_data/oversampling/training_data.npz"
LLM_NPZ   = "../../test_data/oversampling/os_LLM.npz"
SMOTE_NPZ = "../../test_data/oversampling/os_SMOTE.npz"

OUT_STATS_CSV = "pytanie3B_simavg_stats_smote_vs_llm.csv"
OUT_ESCAPES_CSV = "pytanie3B_escape_rate_smote_vs_llm.csv"
OUT_BOXPLOT = "pytanie3B_simavg_boxplot_smote_vs_llm.png"
OUT_HIST = "pytanie3B_simavg_hist_smote_vs_llm.png"

def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0, 0])

def load_training_maps():
    train = np.load(TRAIN_NPZ, allow_pickle=True)
    X_train = train["X"].tolist()
    y_train = train["y"].tolist()
    id_train = train["id"].tolist()
    emb_train = train["embeds"]

    id_to_text = {int(i): str(t) for i, t in zip(id_train, X_train)}
    id_to_emb  = {int(i): emb for i, emb in zip(id_train, emb_train)}
    id_to_y    = {int(i): int(y) for i, y in zip(id_train, y_train)}

    emb_arr = np.array(emb_train)
    y_arr = np.array(y_train)
    centroid_not = emb_arr[y_arr == 0].mean(axis=0)
    centroid_off = emb_arr[y_arr == 1].mean(axis=0)

    return id_to_text, id_to_emb, id_to_y, centroid_not, centroid_off

def build_sim_table(os_npz_path: str, label: str, id_to_emb):
    data = np.load(os_npz_path, allow_pickle=True)
    embeds = data["embeds"]
    p1_ids = data["parent1_id"].tolist()
    p2_ids = data["parent2_id"].tolist()

    rows = []
    missing = 0

    for gen_emb, p1, p2 in zip(embeds, p1_ids, p2_ids):
        p1 = int(p1); p2 = int(p2)
        if p1 not in id_to_emb or p2 not in id_to_emb:
            missing += 1
            continue

        sim1 = cos(gen_emb, id_to_emb[p1])
        sim2 = cos(gen_emb, id_to_emb[p2])
        sim_avg = (sim1 + sim2) / 2.0

        rows.append({
            "method": label,
            "sim_to_p1": sim1,
            "sim_to_p2": sim2,
            "sim_avg": sim_avg
        })

    df = pd.DataFrame(rows)
    return df, missing

def stats_row(method: str, values: pd.Series):
    return {
        "method": method,
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }

def escape_rate(embeds: np.ndarray, centroid_not: np.ndarray, centroid_off: np.ndarray):
    escaped = 0
    for e in embeds:
        sim_not = cos(e, centroid_not)
        sim_off = cos(e, centroid_off)
        if sim_not > sim_off:
            escaped += 1
    return escaped, 100.0 * escaped / len(embeds)


id_to_text, id_to_emb, id_to_y, centroid_not, centroid_off = load_training_maps()
print(f"Loaded training: {len(id_to_emb)} embeddings")

df_llm, miss_llm = build_sim_table(LLM_NPZ, "LLM", id_to_emb)
df_smote, miss_smote = build_sim_table(SMOTE_NPZ, "SMOTE", id_to_emb)

print(f"LLM rows: {len(df_llm)} (missing parents skipped: {miss_llm})")
print(f"SMOTE rows: {len(df_smote)} (missing parents skipped: {miss_smote})")

if len(df_llm) == 0 or len(df_smote) == 0:
    raise RuntimeError("Brak danych w jednej z metod – sprawdź czy ID rodziców pasują do training_data.npz")

stats = pd.DataFrame([
    stats_row("LLM", df_llm["sim_avg"]),
    stats_row("SMOTE", df_smote["sim_avg"])
])
stats.to_csv(OUT_STATS_CSV, index=False)
print(f"Saved stats table: {OUT_STATS_CSV}")
print(stats.to_string(index=False))

plt.figure(figsize=(7, 4))

data = [df_smote["sim_avg"].values, df_llm["sim_avg"].values]
labels = ["SMOTE", "LLM"]

bp = plt.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    showfliers=True
)

for patch in bp["boxes"]:
    patch.set_alpha(0.35)

rng = np.random.default_rng(42)
for i, vals in enumerate(data, start=1):
    vals = np.array(vals)
    if len(vals) > 1500:
        vals = rng.choice(vals, size=1500, replace=False)
    x_jit = rng.normal(loc=i, scale=0.06, size=len(vals))
    plt.scatter(x_jit, vals, alpha=0.15, s=8)

plt.ylabel("sim_avg = avg cosine(generated, parent1/parent2)")
plt.title("Rozkład sim_avg: SMOTE vs LLM (embedding space)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_BOXPLOT, dpi=200)
plt.close()
print(f"Saved boxplot: {OUT_BOXPLOT}")

plt.figure(figsize=(7, 4))
plt.hist(df_smote["sim_avg"], bins=30, alpha=0.6, label="SMOTE")
plt.hist(df_llm["sim_avg"], bins=30, alpha=0.6, label="LLM")
plt.xlabel("sim_avg")
plt.ylabel("Liczba próbek")
plt.title("Histogram sim_avg: SMOTE vs LLM")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_HIST, dpi=200)
plt.close()
print(f"Saved histogram: {OUT_HIST}")

llm_embeds = np.load(LLM_NPZ, allow_pickle=True)["embeds"]
smote_embeds = np.load(SMOTE_NPZ, allow_pickle=True)["embeds"]

llm_esc, llm_esc_pct = escape_rate(llm_embeds, centroid_not, centroid_off)
sm_esc, sm_esc_pct = escape_rate(smote_embeds, centroid_not, centroid_off)

esc_df = pd.DataFrame([
    {"method": "LLM", "escaped_count": int(llm_esc), "escaped_percent": float(llm_esc_pct)},
    {"method": "SMOTE", "escaped_count": int(sm_esc), "escaped_percent": float(sm_esc_pct)},
])
esc_df.to_csv(OUT_ESCAPES_CSV, index=False)
print(f"Saved escape-rate table: {OUT_ESCAPES_CSV}")
print(esc_df.to_string(index=False))
