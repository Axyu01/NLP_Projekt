from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

texts = [
    "Cat is sitting on a fence.",
    "Cat is drinking on a fence.",
    "Cats like to eat human food."
]

embeddings = model.encode(texts)

print("Rozmiar embeddingu:", embeddings.shape)

sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])
sim_2_3 = util.cos_sim(embeddings[1], embeddings[2])

print(f"Podobieństwo 1↔2: {sim_1_2.item():.3f}")
print(f"Podobieństwo 1↔3: {sim_1_3.item():.3f}")
print(f"Podobieństwo 2↔3: {sim_2_3.item():.3f}")
