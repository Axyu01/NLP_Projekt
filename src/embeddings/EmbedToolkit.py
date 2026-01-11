import torch
import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from src.config import CONSTANTS as CONST
import pandas as pd
import os

class EmbedToolkit:
    def __init__(self,INIT_GPT = True,INIT_ENCODER = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(INIT_GPT):
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

        # Sentence Transformer – also moved to GPU if possible
        if(INIT_ENCODER):
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        print("Using device:", self.device)
    

    def rand_frag_remove(self,sentence,MAX_CHAR_REMOVE = 10):
        LEN = len(sentence)
        if LEN<=1:
            return sentence
        start = random.randint(0, LEN - 1)
        end = random.randint(start+1, min(start+MAX_CHAR_REMOVE,LEN))

        return sentence[:start]+sentence[end:]
    
    def rand_frag_add(self,input_ids,greedy_probability = 0.2,MAX_TOKEN_ADD = 5):
        model = self.model
        tokenizer = self.tokenizer
        # random mutation range
        LEN = len(input_ids[0])
        start = token_add = 0

        start = random.randint(0, LEN - 1)
        token_add = random.randint(1, MAX_TOKEN_ADD)

        # Move inputs to GPU
        input_ids = input_ids.to(self.device)

        prefix_tokens = input_ids[:, :start]
        suffix_tokens = input_ids[:, start:]

        gpt_input = torch.cat([suffix_tokens, prefix_tokens], dim=1)

        greedy = random.random() <= greedy_probability
        gen_output = model.generate(
            gpt_input,
            max_length=gpt_input.shape[1] + token_add,
            do_sample=not greedy,
            temperature=1.6,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )


        generated_tokens = gen_output[:, gpt_input.shape[1]:]

        final_ids = torch.cat([prefix_tokens, generated_tokens, suffix_tokens], dim=1)

        final_sentence = tokenizer.decode(final_ids[0].cpu(), skip_special_tokens=True)

        #print("Modified sentence:", final_sentence)
        return final_sentence
    
    def generate(self,input_ids,greedy_probability = 0.2,MAX_TOKEN_GEN = 5):
        model = self.model
        tokenizer = self.tokenizer
        LEN = len(input_ids[0])
        if LEN <=1:
            return tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=True) 
        start = random.randint(0, LEN - 1)
        end = random.randint(start+1, min(start+MAX_TOKEN_GEN,LEN))

        input_ids = input_ids.to(self.device)

        prefix_tokens = input_ids[:, :start]
        fragment_tokens = input_ids[:, start:end]
        suffix_tokens = input_ids[:, end:]

        gpt_input = torch.cat([suffix_tokens, prefix_tokens], dim=1)

        greedy = random.random() <= greedy_probability
        gen_output = model.generate(
            gpt_input,
            max_length=gpt_input.shape[1] + (end - start),
            do_sample=not greedy,
            temperature=1.6,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_tokens = gen_output[:, gpt_input.shape[1]:]

        final_ids = torch.cat([prefix_tokens, generated_tokens, suffix_tokens], dim=1)

        final_sentence = tokenizer.decode(final_ids[0].cpu(), skip_special_tokens=True)

        #print("Modified sentence:", final_sentence)
        return final_sentence


    def embedding(self,text):
        encoder = self.encoder
        emb = encoder.encode(text, convert_to_tensor=True)
        return emb.to(self.device)


    def evaluate(self, solution, target_embed):
        if isinstance(solution, str):
            solution_embed = self.embedding(solution)
        else:
            solution_embed = solution

        return util.cos_sim(solution_embed, target_embed)


    def search(self,target_embed,EPOCHS = 100,NEIGHBOR_SEARCH_NUM = 50,init_sentence = "I love cats and I'm proud of it with all my heart!",DEBUG = False):
        model = self.model
        encoder = self.encoder
        tokenizer = self.tokenizer
        device = self.device

        sentence = init_sentence
        sentence_eval = self.evaluate(sentence, target_embed)

        for i in range(EPOCHS):
            if(DEBUG):
                print("EPOCH:", i)

            for remove in range(1):   
                solutions = []
                evaluations = []
                for s in range(NEIGHBOR_SEARCH_NUM//1):
                    # create neighbor sentence
                    solution = self.rand_frag_remove(sentence)

                    solutions.append(solution)
                    evaluations.append(self.evaluate(solution, target_embed))

                # choose best neighbor
                for s in range(len(solutions)):
                    if sentence_eval < evaluations[s]:
                        sentence = solutions[s]
                        sentence_eval = evaluations[s]
                        if(DEBUG):
                            print("NEW BEST r:", sentence)
            
            input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
            solutions = []
            evaluations = []
            for s in range(NEIGHBOR_SEARCH_NUM):
                # create neighbor sentence
                solution = self.generate(input_ids)

                solutions.append(solution)
                evaluations.append(self.evaluate(solution, target_embed))

                # create neighbor sentence
                solution = self.rand_frag_add(input_ids)

                solutions.append(solution)
                evaluations.append(self.evaluate(solution, target_embed))

            # choose best neighbor
            for s in range(len(solutions)):
                if sentence_eval < evaluations[s]:
                    sentence = solutions[s]
                    sentence_eval = evaluations[s]
                    if(DEBUG):
                        print("NEW BEST:", sentence)
        if(DEBUG):
            print("FINAL:", sentence)
        return sentence
def read_entry(index):
    base_path = "../../dataset"
    train_path = os.path.join(base_path, "olid-training-v1.0.tsv")

    train_df = pd.read_csv(train_path, sep="\t")
    train_df = train_df[['tweet', 'subtask_a']].dropna()

    label_map = {"NOT": 0, "OFF": 1}
    labels = train_df['subtask_a'].map(label_map).tolist()
    texts = train_df['tweet'].tolist()

    print("Przykład tweeta:", texts[index])
    print("Etykieta:", labels[index])

    return texts[index]

if __name__ == "__main__":
    toolkit = EmbedToolkit(INIT_GPT=True,INIT_ENCODER=True)
    embeddings = np.load("../../data/processed/embeddings.npy")
    print("shape",embeddings.shape)

    target = "I hate coffee breaks..."
    target_embed = toolkit.embedding(target)

    index = random.randint(0,embeddings.shape[0])
    print("INDEX:",index)
    target_embed = embeddings[index,:]
    read_entry(index)

    
    target_embed = torch.tensor(target_embed, dtype=torch.float32).to(toolkit.device)
    #target_embed.tolist()

    toolkit.search(target_embed=target_embed,EPOCHS = CONST.REC_EPOCHS,NEIGHBOR_SEARCH_NUM = CONST.REC_NEIGHBORS,init_sentence="I love cats and I'm proud of it with all my heart!",DEBUG=True)


