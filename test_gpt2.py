import torch
import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util

# ======================
#   DEVICE SETUP
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
#   FUNCTIONS
# ======================

def generate(start, end, input_ids, model, tokenizer,gready_probabilty = 0.2):
    # Move inputs to GPU
    input_ids = input_ids.to(device)

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


def embedding(text, encoder):
    emb = encoder.encode(text, convert_to_tensor=True)
    return emb.to(device)


def evaluate(solution, target_embed, encoder):
    solution_embed = embedding(solution, encoder)
    return util.cos_sim(solution_embed, target_embed)

def search(target_embed,model,encoder,tokenizer,EPOCHS = 100,NEIGBOOR_SEARCH_NUM = 50,init_sentance = "I love cats and I'm proud of it wtih all my heart!",DEBUG = False):

    sentence = init_sentance
    sentance_eval = evaluate(sentence, target_embed, encoder)

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
            solution = generate(start, end, input_ids, model, tokenizer)

            solutions.append(solution)
            evaluations.append(evaluate(solution, target_embed, encoder))

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



# ======================
#   LOAD MODELS ON GPU
# ======================

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Sentence Transformer – also moved to GPU if possible
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

embeddings = np.load("smote_embeddings.npy")
print("shape",embeddings.shape)

#target = "I hate coffee breaks..."
#target_embed = embedding(target, encoder)

target_embed = embeddings[random.randint(0,embeddings.shape[0]),:]
target_embed = torch.tensor(target_embed, dtype=torch.float32).to(device)
#target_embed.tolist()

search(target_embed=target_embed,model=model,tokenizer=tokenizer,encoder=encoder,DEBUG=True,init_sentance="Alpha Beta Gamma One Two Three I You Are free funny gamer moment Everytime")


