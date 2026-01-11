from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

text_a = "The cat is sitting on the fence."
text_b = "A small kitten rests on the garden wall."

emb_a = model.encode(text_a, convert_to_tensor=True)
emb_b = model.encode(text_b, convert_to_tensor=True)

base_sim = util.cos_sim(emb_a, emb_b).item()
print(f"Original cosine similarity (A–B): {base_sim:.3f}")

alphas = np.linspace(0, 1, 6)  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
rows = []

for alpha in alphas:
    emb_interp = (1 - alpha) * emb_a + alpha * emb_b
    sim_to_a = util.cos_sim(emb_interp, emb_a).item()
    sim_to_b = util.cos_sim(emb_interp, emb_b).item()
    rows.append({"alpha": alpha, "similarity_to_A": sim_to_a, "similarity_to_B": sim_to_b})

df = pd.DataFrame(rows)
print("\nInterpolation results:")
print(df)

#df.to_csv("interpolation_results.csv", index=False)
