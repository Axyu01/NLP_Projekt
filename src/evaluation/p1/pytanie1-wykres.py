import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

PKL_PATH = "../../test_data/reconstruction/reconstruction.pkl"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

original_embed = np.array(data["original_embed"]).reshape(1, -1)

cosines = []

for trial in data["reconstruction_trials"]:
    end_embed = np.array(trial["end_embed"]).reshape(1, -1)
    sim = cosine_similarity(original_embed, end_embed)[0, 0]
    cosines.append(sim)

cosines = np.array(cosines)

print("Cosine similarities:", np.round(cosines, 4))
print("Mean:", cosines.mean())
print("Std:", cosines.std())

plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)

plt.boxplot(
    cosines,
    vert=True,
    widths=0.4,
    patch_artist=True,
    boxprops=dict(
        facecolor="lightgray",
        alpha=0.4,
        edgecolor="black"
    ),
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black")
)

x_jitter = np.random.normal(1, 0.04, size=len(cosines))
plt.scatter(
    x_jitter,
    cosines,
    color="black",
    alpha=0.85,
    zorder=3
)

plt.ylabel("Cosine similarity")
plt.xticks([1], ["Reconstruction trials"])
plt.title("Boxplot")

plt.subplot(1, 2, 2)

plt.hist(
    cosines,
    bins=5,
    edgecolor="black",
    alpha=0.8
)

plt.xlabel("Cosine similarity")
plt.ylabel("Number of trials")
plt.title("Histogram")


plt.suptitle(
    "Cosine similarity distribution\n(original vs reconstructed embeddings)",
    fontsize=12
)

plt.tight_layout()
plt.savefig("pytanie1_cosine_boxplot_histogram.png", dpi=300)

