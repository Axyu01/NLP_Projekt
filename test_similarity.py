from sentence_transformers import SentenceTransformer

# Ładujemy model (pobrać się powinien automatycznie z Hugging Face)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Testowe zdania
texts = [
    "Cat is sitting on a fance.",
    "Cat is drinking on a fance.",
    "Cats like to eat human food."
]

# Zamiana tekstów na embeddingi
embeddings = model.encode(texts)

print("Rozmiar embeddingu:", embeddings.shape)

from sentence_transformers import util

# podobieństwo między 1. a 2. zdaniem
sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])
sim_2_3 = util.cos_sim(embeddings[1], embeddings[2])

print(f"Podobieństwo 1↔2: {sim_1_2.item():.3f}")
print(f"Podobieństwo 1↔3: {sim_1_3.item():.3f}")
print(f"Podobieństwo 2↔3: {sim_2_3.item():.3f}")
