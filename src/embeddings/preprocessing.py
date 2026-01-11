import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


base_path = "../../dataset"
train_path = os.path.join(base_path, "olid-training-v1.0.tsv")

train_df = pd.read_csv(train_path, sep="\t")
train_df = train_df[['tweet', 'subtask_a']].dropna()

label_map = {"NOT": 0, "OFF": 1}
labels = train_df['subtask_a'].map(label_map).tolist()
texts = train_df['tweet'].tolist()

print("Przykład tweeta:", texts[0])
print("Etykieta:", labels[0])

print("\nŁadowanie modelu MiniLM...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Generowanie embeddingów...")
embeddings = model.encode(texts, show_progress_bar=True)

print("Embeddingi mają kształt:", embeddings.shape)

np.save("../../data/processed/embeddings.npy", embeddings)
np.save("../../data/processed/labels.npy", np.array(labels))

print("Zapisano embeddings.npy i labels.npy")

print("\nObliczanie macierzy podobieństw...")
sim_matrix = cosine_similarity(embeddings)

print("Średnia podobieństw:", sim_matrix.mean())
print("Mediana podobieństw:", np.median(sim_matrix))


i = 0 
closest = np.argsort(-sim_matrix[i])[1]

print("\n=== Przykład najbliższego sąsiada ===")
print("Tweet:", texts[i])
print("Najbardziej podobny:", texts[closest])
print("Podobieństwo:", sim_matrix[i, closest])


X_2d = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="coolwarm", s=8)
plt.title("Reprezentacja embeddingów MiniLM (t-SNE)")
plt.xlabel("Wymiar 1")
plt.ylabel("Wymiar 2")
plt.colorbar(label="Label (0=NOT, 1=OFF)")
plt.show()
