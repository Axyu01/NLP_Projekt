import torch
import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util

# ======================
#   DEVICE SETUP
# ======================
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
    

    # ======================
    #   FUNCTIONS
    # ======================

    def generate(self,start, end, input_ids,gready_probabilty = 0.2):
        model = self.model
        tokenizer = self.tokenizer
        # Move inputs to GPU
        input_ids = input_ids.to(self.device)

        # --- Split tokens ---
        prefix_tokens = input_ids[:, :start]
        fragment_tokens = input_ids[:, start:end]
        suffix_tokens = input_ids[:, end:]

        # --- GPT-2 input = suffix + prefix ---
        gpt_input = torch.cat([suffix_tokens, prefix_tokens], dim=1)

        # --- Generate replacement for fragment ---
        gready = random.random() <= gready_probabilty
        gen_output = model.generate(
            gpt_input,
            max_length=gpt_input.shape[1] + (end - start),
            do_sample=not gready,
            temperature=1.6,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

        # Only take newly generated tokens
        generated_tokens = gen_output[:, gpt_input.shape[1]:]

        # --- Reassemble final sentence ---
        final_ids = torch.cat([prefix_tokens, generated_tokens, suffix_tokens], dim=1)

        # Move back to CPU before decoding
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


    def search(self,target_embed,EPOCHS = 100,NEIGBOOR_SEARCH_NUM = 50,init_sentance = "I love cats and I'm proud of it wtih all my heart!",DEBUG = False):
        model = self.model
        encoder = self.encoder
        tokenizer = self.tokenizer
        device = self.device

        sentence = init_sentance
        sentance_eval = self.evaluate(sentence, target_embed)

        for i in range(EPOCHS):
            if(DEBUG):
                print("EPOCH:", i)
            input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
            LEN = len(input_ids[0])

            solutions = []
            evaluations = []

            for s in range(NEIGBOOR_SEARCH_NUM):

                # random mutation range
                start = end = 0
                while start == end:
                    start = random.randint(0, LEN - 1)
                    end = random.randint(0, LEN - 1)

                if start > end:
                    start, end = end, start

                # create neighbor sentence
                solution = self.generate(start, end, input_ids)

                solutions.append(solution)
                evaluations.append(self.evaluate(solution, target_embed))

            # choose best neighbor
            for s in range(NEIGBOOR_SEARCH_NUM):
                if sentance_eval < evaluations[s]:
                    sentence = solutions[s]
                    sentance_eval = evaluations[s]
                    if(DEBUG):
                        print("NEW BEST:", sentence)
        if(DEBUG):
            print("FINAL:", sentence)
        return sentence

if __name__ == "__main__":
    toolkit = EmbedToolkit(INIT_GPT=True,INIT_ENCODER=True)
    embeddings = np.load("embeddings.npy")
    print("shape",embeddings.shape)

    #target = "I hate coffee breaks..."
    #target_embed = toolkit.embedding(target)

    target_embed = embeddings[random.randint(0,embeddings.shape[0]),:]
    target_embed = torch.tensor(target_embed, dtype=torch.float32).to(toolkit.device)
    #target_embed.tolist()

    toolkit.search(target_embed=target_embed,EPOCHS = 10,NEIGBOOR_SEARCH_NUM = 10,init_sentance="I love cats and I'm proud of it wtih all my heart!",DEBUG=True)


