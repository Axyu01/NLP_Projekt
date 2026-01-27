import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_NPZ = "../../test_data/oversampling/training_data.npz"
LLM_NPZ   = "../../test_data/oversampling/os_LLM.npz"
SMOTE_NPZ = "../../test_data/oversampling/os_SMOTE.npz"

N_EXAMPLES_PER_METHOD = 5
MAX_CHARS = 140

OUT_CSV = "pytanie3_c_language_examples.csv"
OUT_TXT = "pytanie3_c_language_examples_full.txt"


def shorten(text: str, max_chars: int = 140) -> str:
    if text is None:
        return ""
    t = str(text).replace("\n", " ").strip()
    return t if len(t) <= max_chars else t[:max_chars] + "…"


def cos(a, b) -> float:
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0, 0])

train = np.load(TRAIN_NPZ, allow_pickle=True)
X_train = train["X"].tolist()
id_train = train["id"].tolist()
emb_train = train["embeds"]

id_to_text = {int(i): str(t) for i, t in zip(id_train, X_train)}
id_to_emb  = {int(i): emb for i, emb in zip(id_train, emb_train)}

print(f"Loaded training: {len(id_to_text)} rows")

llm = np.load(LLM_NPZ, allow_pickle=True)
llm_samples = llm["samples"].tolist()
llm_embeds = llm["embeds"]
llm_p1 = llm["parent1_id"].tolist()
llm_p2 = llm["parent2_id"].tolist()


smote = np.load(SMOTE_NPZ, allow_pickle=True)
smote_embeds = smote["embeds"]
smote_p1 = smote["parent1_id"].tolist()
smote_p2 = smote["parent2_id"].tolist()


llm_rows = []
for gen_text, gen_emb, p1, p2 in zip(llm_samples, llm_embeds, llm_p1, llm_p2):
    p1 = int(p1); p2 = int(p2)
    if p1 not in id_to_text or p2 not in id_to_text:
        continue
    if p1 not in id_to_emb or p2 not in id_to_emb:
        continue

    sim1 = cos(gen_emb, id_to_emb[p1])
    sim2 = cos(gen_emb, id_to_emb[p2])
    sim_avg = (sim1 + sim2) / 2

    llm_rows.append({
        "method": "LLM",
        "parent1_id": p1,
        "parent1_text": id_to_text[p1],
        "parent2_id": p2,
        "parent2_text": id_to_text[p2],
        "synthetic_text": str(gen_text),
        "sim_to_p1": sim1,
        "sim_to_p2": sim2,
        "sim_avg": sim_avg
    })

df_llm = pd.DataFrame(llm_rows).sort_values("sim_avg", ascending=False)


def pick_diverse(df, k):
    if len(df) <= k:
        return df.copy()
    top = df.head(k // 3)
    bot = df.tail(k // 3)
    mid_n = k - len(top) - len(bot)
    mid_start = (len(df) - mid_n) // 2
    mid = df.iloc[mid_start:mid_start + mid_n]
    return pd.concat([top, mid, bot], axis=0).drop_duplicates().reset_index(drop=True)

df_llm_pick = pick_diverse(df_llm, N_EXAMPLES_PER_METHOD)


smote_rows = []
for gen_emb, p1, p2 in zip(smote_embeds, smote_p1, smote_p2):
    p1 = int(p1); p2 = int(p2)
    if p1 not in id_to_text or p2 not in id_to_text:
        continue
    if p1 not in id_to_emb or p2 not in id_to_emb:
        continue

    sim1 = cos(gen_emb, id_to_emb[p1])
    sim2 = cos(gen_emb, id_to_emb[p2])
    sim_avg = (sim1 + sim2) / 2

    smote_rows.append({
        "method": "SMOTE",
        "parent1_id": p1,
        "parent1_text": id_to_text[p1],
        "parent2_id": p2,
        "parent2_text": id_to_text[p2],
        "synthetic_text": f"[SMOTE_SYNTHETIC_FROM_{p1}_{p2}]",
        "sim_to_p1": sim1,
        "sim_to_p2": sim2,
        "sim_avg": sim_avg
    })

df_smote = pd.DataFrame(smote_rows).sort_values("sim_avg", ascending=False)
df_smote_pick = pick_diverse(df_smote, N_EXAMPLES_PER_METHOD)

df_combined = pd.concat([df_llm_pick, df_smote_pick], axis=0).reset_index(drop=True)

df_report = df_combined.copy()
df_report["parent1_text"] = df_report["parent1_text"].apply(lambda t: shorten(t, MAX_CHARS))
df_report["parent2_text"] = df_report["parent2_text"].apply(lambda t: shorten(t, MAX_CHARS))
df_report["synthetic_text"] = df_report["synthetic_text"].apply(lambda t: shorten(t, MAX_CHARS))

df_report = df_report[
    ["method", "parent1_id", "parent1_text", "parent2_id", "parent2_text", "synthetic_text", "sim_to_p1", "sim_to_p2", "sim_avg"]
].sort_values(["method", "sim_avg"], ascending=[True, False])

df_report.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")


with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("PYTANIE 3C — PORÓWNANIE JĘZYKOWE: LLM vs SMOTE (PEŁNE TEKSTY)\n\n")
    f.write("UWAGA: SMOTE nie generuje tekstu — tylko embedding (interpolacja). W tabeli SMOTE pokazujemy placeholder.\n\n")

    for _, row in df_combined.iterrows():
        f.write("------------------------------------------------------------\n")
        f.write(f"METHOD: {row['method']} | sim_to_p1={row['sim_to_p1']:.6f} | sim_to_p2={row['sim_to_p2']:.6f} | sim_avg={row['sim_avg']:.6f}\n")
        f.write(f"PARENT1 (id={row['parent1_id']}):\n{row['parent1_text']}\n\n")
        f.write(f"PARENT2 (id={row['parent2_id']}):\n{row['parent2_text']}\n\n")
        f.write("SYNTHETIC:\n")
        f.write(f"{row['synthetic_text']}\n\n")

print(f"Saved: {OUT_TXT}")

pd.set_option("display.max_colwidth", 60)
print("\n=== TABELA DO RAPORTU (skrócona) ===")
print(df_report.to_string(index=False))
