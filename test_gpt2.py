import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer,util
def generate(start,end,input_ids,model,tokenizer):
    # --- Split tokens ---
    prefix_tokens = input_ids[:, :start]
    fragment_tokens = input_ids[:, start:end]
    suffix_tokens = input_ids[:, end:]

    # --- GPT-2 input = prefix + suffix ---
    gpt_input = torch.cat([suffix_tokens,prefix_tokens], dim=1)

    # --- Generate replacement for fragment ---
    gen_output = model.generate(
        gpt_input,
        max_length=gpt_input.shape[1] + (end - start),
        do_sample=True,
        temperature=0.7,
        top_k=30,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    # Only take newly generated tokens
    generated_tokens = gen_output[:, gpt_input.shape[1]:]

    # --- Reassemble final sentence ---
    final_ids = torch.cat([prefix_tokens, generated_tokens, suffix_tokens], dim=1)

    # --- Decode ---
    final_sentence = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    #input_sentance = tokenizer.decode(gpt_input[0], skip_special_tokens=True)
    #print("Original sentence:", sentence)
    print("Modified sentence:", final_sentence)
    #print("Input sentence:", input_sentance)
    return final_sentence
def embedding(target,encoder):
    return encoder.encode(target)
def evaluate(solution,target_embed,encoder):
    solution_embed = embedding(solution,encoder)
    return util.cos_sim(solution_embed,target_embed)

# --- Initialize GPT-2 ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")  # .to("cuda") if available
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

EPOCHS = 100
NEIGBOOR_SEARCH_NUM = 30

# --- Example ---
target = "I enjoy cars and I'm depressed"
target_embed = embedding(target,encoder)

#start sentance
sentence = "I love cats and I'm proud of it!"
sentance_eval = evaluate(sentence,target_embed,encoder)

for i in range(EPOCHS):
    print("EPOCH:",i)
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    LEN = len(input_ids[0])

    solutions = []
    evaluations = []
    for s in range(NEIGBOOR_SEARCH_NUM):
        start = 0
        end = 0
        while start == end:
            start = random.randint(0,LEN-1)
            end = random.randint(0,LEN-1)

        if start > end:
            temp = start
            start = end
            end = temp

        solution = generate(start,end,input_ids,model,tokenizer)
        solutions.append(solution)
        evaluations.append(evaluate(solution,target_embed,encoder))

    for s in range(NEIGBOOR_SEARCH_NUM):
        if sentance_eval < evaluations[s]:
            sentence = solutions[s]
            sentance_eval = evaluations[s]

print(sentence)

