from src.config import CONSTANTS as CONST
from src.embeddings.EmbedToolkit import EmbedToolkit,read_entry
import pandas as pd
import numpy as np
import random
import torch

def save_test_data():
    pass
def load__test_data():
    pass


if __name__ == "__main__":
    toolkit = EmbedToolkit(INIT_GPT=True,INIT_ENCODER=True)
    embeddings = np.load("embeddings.npy")
    print("shape",embeddings.shape)

    target = "I hate coffee breaks..."
    target_embed = toolkit.embedding(target)

    index = random.randint(0,embeddings.shape[0])
    print("INDEX:",index)
    target_embed = embeddings[index,:]
    target = read_entry(index)


    target_embed = torch.tensor(target_embed, dtype=torch.float32).to(toolkit.device)
    #target_embed.tolist()

    end_resault = toolkit.search(target_embed=target_embed,EPOCHS = CONST.REC_EPOCHS,NEIGHBOR_SEARCH_NUM = CONST.REC_NEIGHBORS,init_sentence="I love cats and I'm proud of it with all my heart!",DEBUG=True)

    print("Target:",target)
    print("Result:",end_resault)
