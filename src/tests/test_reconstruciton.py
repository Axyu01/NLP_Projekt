from sklearn.metrics.pairwise import cosine_similarity

from src.config import CONSTANTS as CONST
from src.embeddings.EmbedToolkit import EmbedToolkit,read_entry
import pandas as pd
import numpy as np
import random
import torch
import os
import pickle

def save_test_data(data,path=CONST.PATH_TEST_DATA_RECONSTRUCTION):
    if isinstance(data.get("original_embed"), torch.Tensor):
        data["original_embed"] = data["original_embed"].detach().cpu().numpy()

    for trial in data["reconstruction_trials"]:
        if isinstance(trial.get("end_embed"), torch.Tensor):
            trial["end_embed"] = trial["end_embed"].detach().cpu().numpy()

    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_test_data(path=CONST.PATH_TEST_DATA_RECONSTRUCTION):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def analyze_reconstruction(data):
    orig = data["original_embed"].reshape(1, -1)

    sims = []
    texts = []

    for trial in data["reconstruction_trials"]:
        emb = trial["end_embed"].reshape(1, -1)
        sim = cosine_similarity(orig, emb)[0, 0]
        sims.append(sim)
        texts.append(trial["end_result"])

    sims = np.array(sims)

    best_idx = np.argmax(sims)
    worst_idx = np.argmin(sims)

    row = {
        "original_text": data["original_text"],
        "best_text": texts[best_idx],
        "best_cosine": sims[best_idx],
        "worst_text": texts[worst_idx],
        "worst_cosine": sims[worst_idx],
        "mean_cosine": sims.mean(),
        "std_cosine": sims.std()
    }

    return pd.DataFrame([row])


if __name__ == "__main__":
    toolkit = EmbedToolkit(INIT_GPT=True,INIT_ENCODER=True)
    embeddings = np.load("../../data/processed/embeddings.npy")
    print("shape",embeddings.shape)

    index = random.randint(0,embeddings.shape[0]-1)
    print("INDEX:",index)
    target_embed = embeddings[index,:]
    target = read_entry(index)
    #target = "I hate coffee breaks..."
    #target_embed = toolkit.embedding(target)

    data = {
        "original_text":target,
        "original_index":index,
        "original_embed":target_embed,
        "reconstruction_trials": []
        }

    target_embed = torch.tensor(target_embed, dtype=torch.float32).to(toolkit.device)
    for i in range(CONST.REC_TRIALS):
        end_result = toolkit.search(target_embed=target_embed,EPOCHS = CONST.REC_EPOCHS,NEIGHBOR_SEARCH_NUM = CONST.REC_NEIGHBORS,init_sentence="I love cats and I'm proud of it with all my heart!",DEBUG=True)
        end_embed = toolkit.embedding(end_result)
        data["reconstruction_trials"].append({"end_result":end_result,"end_embed":end_embed})
        print("Target:",target)
        print("Result:",end_result)
    save_test_data(data=data)
    #data_l = load_test_data()
    #print(data_l)

    df = analyze_reconstruction(data)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/reconstruction_summary.csv", index=False)
    print("\nTabela rekonstrukcji:")
    print(df)
