import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import random

embeddings = np.load("../../data/processed/embeddings.npy")
labels = np.load("../../data/processed/labels.npy")

print("Załadowano:", embeddings.shape, labels.shape)

# klasa mniejszościowa = OFF = 1
minority_class = 1

minority_idx = np.where(labels == minority_class)[0]
majority_idx = np.where(labels == 0)[0]

emb_min = embeddings[minority_idx]
emb_maj = embeddings[majority_idx]

print(f"Liczba OFF: {len(emb_min)}, liczba NOT: {len(emb_maj)}")


def generate_smote_samples(embeddings_minority, num_samples, k=5):
    """
    Generuje syntetyczne embeddingi metodą SMOTE.

    Algorytm:
        - dla losowego punktu klasy mniejszościowej
        - wybiera jednego z k najbliższych sąsiadów
        - interpoluje liniowo między punktami

    Args:
        embeddings_minority (np.ndarray):
            Embeddingi klasy mniejszościowej, shape (N, D)
        num_samples (int):
            Liczba syntetycznych punktów do wygenerowania
        k (int):
            Liczba sąsiadów branych pod uwagę (SMOTE)

    Returns:
        np.ndarray:
            Tablica syntetycznych embeddingów, shape (num_samples, D)
    """
    nn = NearestNeighbors(n_neighbors=k+1).fit(embeddings_minority)
    distances, indices = nn.kneighbors(embeddings_minority)

    synthetic = []

    for _ in range(num_samples):
        i = random.randint(0, len(embeddings_minority)-1)

        neighbor = random.choice(indices[i][1:])  
        diff = embeddings_minority[neighbor] - embeddings_minority[i]

        lam = np.random.rand()
        new_point = embeddings_minority[i] + lam * diff
        synthetic.append(new_point)

    return np.array(synthetic)


def generate_mixup_samples(embeddings_minority, num_samples):
    """
    Generuje syntetyczne embeddingi metodą MIXUP.

    Algorytm:
        - losuje parę punktów klasy mniejszościowej
        - tworzy ich wypukłą kombinację liniową

    Args:
        embeddings_minority (np.ndarray):
            Embeddingi klasy mniejszościowej, shape (N, D)
        num_samples (int):
            Liczba syntetycznych punktów do wygenerowania

    Returns:
        np.ndarray:
            Tablica syntetycznych embeddingów, shape (num_samples, D)
    """
    synthetic = []
    n = len(embeddings_minority)

    for _ in range(num_samples):
        i, j = np.random.choice(n, 2, replace=False)
        alpha = np.random.rand()
        new_point = alpha * embeddings_minority[i] + (1 - alpha) * embeddings_minority[j]
        synthetic.append(new_point)

    return np.array(synthetic)

N_SYNTH = 2000   # ile chcesz wygenerować

smote_emb = generate_smote_samples(emb_min, N_SYNTH, k=5)
mixup_emb = generate_mixup_samples(emb_min, N_SYNTH)

print("Wygenerowano SMOTE:", smote_emb.shape)
print("Wygenerowano MIXUP:", mixup_emb.shape)


def nearest_original(point, originals):
    """
    Znajduje najbliższy embedding w zbiorze oryginalnym
    na podstawie podobieństwa cosinusowego.

    Args:
        point (np.ndarray):
            Embedding syntetyczny, shape (D,)
        originals (np.ndarray):
            Embeddingi oryginalne, shape (N, D)

    Returns:
        tuple:
            - idx (int): indeks najbliższego punktu
            - sim (float): podobieństwo cosinusowe
    """
    sims = cosine_similarity(point.reshape(1, -1), originals)[0]
    idx = sims.argmax()
    return idx, sims[idx]


def escape_analysis(synthetic_points, original_min, original_maj):
    """
        Analiza "ucieczek" embeddingów syntetycznych.

        Dla każdego punktu syntetycznego:
            - znajduje najbliższy embedding OFF
            - znajduje najbliższy embedding NOT
            - sprawdza, czy punkt jest bliżej NOT niż OFF

        Args:
            synthetic_points (np.ndarray):
                Embeddingi syntetyczne, shape (N, D)
            original_min (np.ndarray):
                Oryginalne embeddingi OFF
            original_maj (np.ndarray):
                Oryginalne embeddingi NOT

        Returns:
            dict:
                Statystyki ucieczek:
                    - escaped_count
                    - escaped_percent
                    - avg_sim_to_OFF
                    - avg_sim_to_NOT
    """
    escaped = 0
    sims_min = []
    sims_maj = []

    for p in synthetic_points:
        _, sim_min = nearest_original(p, original_min)
        _, sim_maj = nearest_original(p, original_maj)

        sims_min.append(sim_min)
        sims_maj.append(sim_maj)

        # jeśli punkt jest bardziej podobny do NOT to uciekł
        if sim_maj > sim_min:
            escaped += 1

    return {
        "escaped_count": escaped,
        "escaped_percent": 100 * escaped / len(synthetic_points),
        "avg_sim_to_OFF": np.mean(sims_min),
        "avg_sim_to_NOT": np.mean(sims_maj),
    }

print("ANALIZA UCIECZEK DLA SMOTE")
esc_smote = escape_analysis(smote_emb, emb_min, emb_maj)
print(esc_smote)

print("ANALIZA UCIECZEK DLA MIXUP")
esc_mixup = escape_analysis(mixup_emb, emb_min, emb_maj)
print(esc_mixup)

np.save("../../data/synthetic/smote_embeddings.npy", smote_emb)
np.save("../../data/synthetic/mixup_embeddings.npy", mixup_emb)

print("\nZapisano smote_embeddings.npy i mixup_embeddings.npy.")

