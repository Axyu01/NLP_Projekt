Helpfull links:
https://arxiv.org/abs/2205.12035?utm_source=chatgpt.com

//Ja uzylem tego
pip install -U sentence-transformers

pip install sentence-transformers torch

Kilka dobrych open-source opcji:

Model	Cechy	Link
sentence-transformers/all-MiniLM-L6-v2	szybki, mały, 384-dim	Hugging Face

intfloat/e5-base-v2	nowszy, lepsze wyniki, 768-dim	HF

nomic-ai/nomic-embed-text-v1	długi kontekst, open weights	HF


## Struktura projektu

├───data
│   ├───processed
│   ├───raw
│   └───synthetic
├───dataset
├───src
│   ├───config
│   ├───embeddings
│   ├───evaluation
│   ├───oversampling
│   └───tests
├───test_data
│   └───oversampling

## Wymagania

- Python 3.10+
- torch
- sentence-transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- groq (dla LLM oversampler)