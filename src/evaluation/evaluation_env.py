import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


"""
BLOK A:
    Analiza embeddingów syntetycznych (SMOTE, MIXUP):
    - porównanie z oryginalnymi embeddingami
    - wykrywanie tzw. "ucieczek" (escape), czyli sytuacji,
      gdy embedding syntetyczny jest bliżej klasy większościowej
      niż mniejszościowej

BLOK B:
    Narzędzia do wyszukiwania podobnych tweetów (k-NN)
    oraz prostej klasyfikacji 1-NN w przestrzeni embeddingów.

Plik może być:
    - importowany jako moduł (funkcje z BLOKU B)
    - uruchamiany jako skrypt (analiza SMOTE/MIXUP)
"""

embeddings = np.load("../../data/processed/embeddings.npy")
labels = np.load("../../data/processed/labels.npy")

smote_emb = np.load("../../data/synthetic/smote_embeddings.npy")
mixup_emb = np.load("../../data/synthetic/mixup_embeddings.npy")

print("Kształt oryginalnych embeddingów:", embeddings.shape)
print("Kształt etykiet:", labels.shape)
print("Kształt SMOTE:", smote_emb.shape)
print("Kształt MIXUP:", mixup_emb.shape)

MINORITY_CLASS = 1   # OFF
MAJORITY_CLASS = 0   # NOT

emb_min = embeddings[labels == MINORITY_CLASS]
emb_maj = embeddings[labels == MAJORITY_CLASS]

print(f"Liczba OFF: {len(emb_min)}, liczba NOT: {len(emb_maj)}")

def nearest_original(point: np.ndarray, originals: np.ndarray):
    """
    Zwraca indeks i podobieństwo (cosinusowe) najbliższego oryginalnego punktu z tablicy `originals`.
    """
    sims = cosine_similarity(point.reshape(1, -1), originals)[0]  # (n,)
    idx = sims.argmax()
    return idx, sims[idx]


def escape_analysis(synth: np.ndarray):
    """
    Dla każdego syntetycznego punktu (np. z SMOTE/MIXUP):
      - liczy podobieństwo do najbliższego OFF (emb_min)
      - liczy podobieństwo do najbliższego NOT (emb_maj)
      - sprawdza, czy 'uciekł' (bardziej podobny do NOT niż do OFF)

    Zwraca:
      - stats: słownik ze statystykami ogólnymi
      - df: DataFrame z wynikami dla każdego punktu
    """
    escaped = 0
    sims_min = []
    sims_maj = []

    for p in synth:
        _, sim_min = nearest_original(p, emb_min)
        _, sim_maj = nearest_original(p, emb_maj)

        sims_min.append(sim_min)
        sims_maj.append(sim_maj)

        # jeśli punkt jest bardziej podobny do NOT niż do OFF -> ucieczka
        if sim_maj > sim_min:
            escaped += 1

    sims_min = np.array(sims_min)
    sims_maj = np.array(sims_maj)

    stats = {
        "escaped_count": int(escaped),
        "escaped_percent": float(100.0 * escaped / len(synth)),
        "avg_sim_to_OFF": float(sims_min.mean()),
        "avg_sim_to_NOT": float(sims_maj.mean()),
    }

    df = pd.DataFrame({
        "sim_to_OFF": sims_min,
        "sim_to_NOT": sims_maj,
        "escaped": sims_maj > sims_min,
    })

    return stats, df


BASE_PATH = "../../dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "olid-training-v1.0.tsv")

print("\nWczytuję tweety z OLID...")
train_df = pd.read_csv(TRAIN_PATH, sep="\t")
train_df = train_df[['tweet', 'subtask_a']].dropna()

label_map = {"NOT": 0, "OFF": 1}
reverse_label_map = {0: "NOT", 1: "OFF"}

texts = train_df["tweet"].tolist()
labels_text = train_df["subtask_a"].map(label_map).to_numpy()

print(f"Liczba tweetów w dataframe: {len(texts)}")
print("Ładuję model MiniLM (do embedowania nowych tekstów)...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> np.ndarray:
    """
    Zamienia pojedynczy tekst na embedding (MiniLM).
    """
    return encoder.encode([text])[0]


def nearest_neighbors(query_emb: np.ndarray, k: int = 5):
    """
    Zwraca indeksy k najbliższych tweetów z OLID oraz odpowiadające im podobieństwa cosinusowe.
    """
    sims = cosine_similarity(query_emb.reshape(1, -1), embeddings)[0]  # (N,)
    idx_sorted = np.argsort(-sims)
    top_idx = idx_sorted[:k]
    top_sims = sims[top_idx]
    return top_idx, top_sims


def classify_1nn(query_emb: np.ndarray) -> int:
    """
    Klasyfikacja 1-NN najbardziej podobny tweet,
    """
    idx, sims = nearest_neighbors(query_emb, k=1)
    nearest_idx = int(idx[0])
    return int(labels[nearest_idx])


def evaluate_text(text: str, k: int = 5) -> pd.DataFrame:
    """
    funkcja pomocnicza:
      - embeduje nowy tekst,
      - zwraca tabelę z k najbliższymi tweetami:
        * similarity
        * tweet
        * label (NOT/OFF)
    """
    q_emb = embed_text(text)
    top_idx, top_sims = nearest_neighbors(q_emb, k=k)

    rows = []
    for i, sim in zip(top_idx, top_sims):
        i = int(i)
        rows.append({
            "similarity": float(sim),
            "tweet": texts[i],
            "label": reverse_label_map[int(labels[i])]
        })

    df = pd.DataFrame(rows)
    return df

def evaluate_text_list(input_texts, k: int = 5, target_text: str | None = None) -> pd.DataFrame:
    """
    Ewaluacja listy tekstów (np. zdań wygenerowanych przez GPT-2).

    Dla każdego tekstu:
      - liczy embedding (MiniLM),
      - znajduje k najbliższych tweetów z OLID,
      - zapisuje podobieństwo i etykietę najbliższego tweeta (1-NN),
      - opcjonalnie liczy podobieństwo do tekstu docelowego (target_text).

    Zwraca DataFrame z kolumnami m.in.:
      text, nearest_tweet, nearest_sim, pred_label, pred_label_name, [sim_to_target]
    """
    rows = []

    target_emb = None
    if target_text is not None:
        target_emb = embed_text(target_text)

    for t in input_texts:
        q_emb = embed_text(t)
        nn_idx, nn_sims = nearest_neighbors(q_emb, k=1)
        idx0 = int(nn_idx[0])
        sim_nn = float(nn_sims[0])
        pred_label = int(labels[idx0])
        pred_label_name = reverse_label_map[pred_label]

        row = {
            "text": t,
            "nearest_tweet": texts[idx0],
            "nearest_sim": sim_nn,
            "pred_label": pred_label,
            "pred_label_name": pred_label_name,
        }

        if target_emb is not None:
            sim_target = float(
                cosine_similarity(
                    q_emb.reshape(1, -1),
                    target_emb.reshape(1, -1)
                )[0, 0]
            )
            row["sim_to_target"] = sim_target

        rows.append(row)

    return pd.DataFrame(rows)



#  Funkcje z BLOKU B używamy z pliku temp.py !!!


if __name__ == "__main__":
    print("\n=== Analiza ucieczek dla SMOTE ===")
    smote_stats, smote_df = escape_analysis(smote_emb)
    print(smote_stats)

    print("\n=== Analiza ucieczek dla MIXUP ===")
    mixup_stats, mixup_df = escape_analysis(mixup_emb)
    print(mixup_stats)

    smote_df.to_csv("smote_escape_details.csv", index=False)
    mixup_df.to_csv("mixup_escape_details.csv", index=False)

    print("\nZapisano smote_escape_details.csv i mixup_escape_details.csv.")
    print("\nFunkcje z BLOKU B (embed_text, nearest_neighbors, classify_1nn, evaluate_text)")
    print("są dostępne po imporcie modułu evaluation_env.")
