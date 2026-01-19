import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

class SMOTEOversampler:
    def __init__(self, k=5, seed=42):
        self.k = k
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def oversample_data(self, X_minority, id_minority, embeds_minority, n_samples=None):
        """
        SMOTE oversampling na embeddingach

        Args:
            X_minority (list[str])     – teksty klasy mniejszościowej
            id_minority (list[int])    – ID oryginalnych próbek
            embeds_minority (list[np.ndarray]) – embeddingi (N, D)
            n_samples (int | None)     – ile próbek wygenerować (domyślnie = tyle co minority)

        Returns:
            samples (list[str])        – placeholder tekstów (SMOTE = embedding-only)
            embeds (list[np.ndarray])  – syntetyczne embeddingi
            parent1_id (list[int])
            parent2_id (list[int])
        """

        embeds = np.array(embeds_minority)

        if n_samples is None:
            n_samples = len(embeds)

        nn = NearestNeighbors(n_neighbors=self.k + 1, metric="cosine")
        nn.fit(embeds)
        _, indices = nn.kneighbors(embeds)

        synthetic_embeds = []
        parent1_ids = []
        parent2_ids = []
        synthetic_texts = []

        for _ in range(n_samples):
            i = random.randint(0, len(embeds) - 1)
            neighbor_idx = random.choice(indices[i][1:])

            emb_i = embeds[i]
            emb_j = embeds[neighbor_idx]

            lam = np.random.rand()
            new_emb = emb_i + lam * (emb_j - emb_i)

            synthetic_embeds.append(new_emb)
            parent1_ids.append(id_minority[i])
            parent2_ids.append(id_minority[neighbor_idx])

            synthetic_texts.append(f"[SMOTE_SYNTHETIC_FROM_{id_minority[i]}_{id_minority[neighbor_idx]}]")

        return (
            synthetic_texts,
            synthetic_embeds,
            parent1_ids,
            parent2_ids,
        )
