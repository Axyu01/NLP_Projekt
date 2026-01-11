import torch
import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from src.config import CONSTANTS as CONST
import pandas as pd
import os


class EmbedToolkit:
    """
        Toolkit do:
        - generowania embeddingów zdań (MiniLM),
        - lokalnych modyfikacji tekstu (mutacje),
        - generowania tekstu GPT-2,
        - iteracyjnego wyszukiwania zdania o maksymalnym podobieństwie
          embeddingowym do zadanego wektora docelowego.

        Klasa wykorzystywana m.in. do:
        - rekonstrukcji tekstu z embeddingu,
        - eksploracyjnego oversamplingu semantycznego.
        """

    def __init__(self, INIT_GPT=True, INIT_ENCODER=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (INIT_GPT):
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

        # Sentence Transformer – also moved to GPU if possible
        if (INIT_ENCODER):
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        print("Using device:", self.device)

    def rand_frag_remove(self, sentence, MAX_CHAR_REMOVE=10):
        """
            Losowo usuwa fragment tekstu (operacja delecji).
            Używane jako lokalna mutacja w przestrzeni tekstowej.

            Parameters
            ----------
            sentence : str
                Wejściowe zdanie.
            MAX_CHAR_REMOVE : int
                Maksymalna liczba znaków do usunięcia.

            Returns
            -------
            str
                Zmutowane zdanie.
            """
        LEN = len(sentence)
        if LEN <= 1:
            return sentence
        start = random.randint(0, LEN - 1)
        end = random.randint(start + 1, min(start + MAX_CHAR_REMOVE, LEN))

        return sentence[:start] + sentence[end:]

    def rand_frag_add(self, input_ids, greedy_probability=0.2, MAX_TOKEN_ADD=5):
        """
            Losowo dodaje fragment wygenerowany przez GPT-2 w losowym miejscu zdania.

            Parameters
            ----------
            input_ids : torch.Tensor
                Tokeny wejściowe zdania.
            greedy_probability : float
                Prawdopodobieństwo użycia greedy decoding (vs sampling).
            MAX_TOKEN_ADD : int
                Maksymalna liczba tokenów do wygenerowania.

            Returns
            -------
            str
                Zmutowane zdanie.
            """
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
        if gpt_input.shape[1] == 0:
            gpt_input = self.random_token_input()

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

        # print("Modified sentence:", final_sentence)
        return final_sentence
    
    def random_token_input(self):
        tokenizer = self.tokenizer

        # pick a random token that is NOT special
        while True:
            token_id = random.randint(0, tokenizer.vocab_size - 1)
            if token_id not in tokenizer.all_special_ids:
                break

        return torch.tensor([[token_id]], device=self.device)
    
    def generate(self, input_ids, greedy_probability=0.2, MAX_TOKEN_GEN=5):
        """
        Zastępuje losowy fragment zdania nowym fragmentem
        wygenerowanym przez GPT-2.

        Jest to mutacja typu „replace”.

        Returns
        -------
        str
            Zmutowane zdanie.
        """
        model = self.model
        tokenizer = self.tokenizer
        LEN = len(input_ids[0])
        if LEN <= 1:
            return tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=True)
        start = random.randint(0, LEN - 1)
        end = random.randint(start + 1, min(start + MAX_TOKEN_GEN, LEN))

        input_ids = input_ids.to(self.device)

        prefix_tokens = input_ids[:, :start]
        fragment_tokens = input_ids[:, start:end]
        suffix_tokens = input_ids[:, end:]

        gpt_input = torch.cat([suffix_tokens, prefix_tokens], dim=1)
        if gpt_input.shape[1] == 0:
            gpt_input = self.random_token_input()

        greedy = random.random() <= greedy_probability
        gen_output = model.generate(
            gpt_input,
            max_length=gpt_input.shape[1] + max((end - start),1),
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

        # print("Modified sentence:", final_sentence)
        return final_sentence

    def embedding(self, text):
        """
        Oblicza embedding zdania (MiniLM).

        Parameters
        ----------
        text : str

        Returns
        -------
        torch.Tensor
            Embedding zdania.
        """
        encoder = self.encoder
        emb = encoder.encode(text, convert_to_tensor=True)
        return emb.to(self.device)

    def evaluate(self, solution, target_embed):
        """
        Liczy podobieństwo cosinusowe między rozwiązaniem a embeddingiem docelowym.

        Parameters
        ----------
        solution : str | torch.Tensor
            Zdanie lub embedding.
        target_embed : torch.Tensor
            Embedding docelowy.

        Returns
        -------
        torch.Tensor
            Cosine similarity.
        """
        if isinstance(solution, str):
            solution_embed = self.embedding(solution)
        else:
            solution_embed = solution

        return util.cos_sim(solution_embed, target_embed)

    def search(self, target_embed, EPOCHS=100, NEIGHBOR_SEARCH_NUM=50,
               init_sentence="I love cats and I'm proud of it with all my heart!", DEBUG=False):
        """
        Iteracyjny algorytm lokalnego przeszukiwania w przestrzeni tekstów.

        Cel:
        znaleźć zdanie, którego embedding jest jak najbardziej podobny do `target_embed`.

        Metoda:
            - mutacje tekstowe (remove / replace / add),
            - greedy hill-climbing w przestrzeni embeddingów.

        Returns
        -------
        str
            Najlepsze znalezione zdanie.
        """
        model = self.model
        encoder = self.encoder
        tokenizer = self.tokenizer
        device = self.device

        sentence = init_sentence
        sentence_eval = self.evaluate(sentence, target_embed)

        for i in range(EPOCHS):
            if (DEBUG):
                print("EPOCH:", i)

            for remove in range(CONST.REC_REMOVE_ITERATIONS):   
                solutions = []
                evaluations = []
                for s in range(NEIGHBOR_SEARCH_NUM):
                    # create neighbor sentence
                    solution = self.rand_frag_remove(sentence,MAX_CHAR_REMOVE=CONST.REC_MAX_CHAR_REMOVE)

                    solutions.append(solution)
                    evaluations.append(self.evaluate(solution, target_embed))

                # choose best neighbor
                for s in range(len(solutions)):
                    if sentence_eval < evaluations[s]:
                        sentence = solutions[s]
                        sentence_eval = evaluations[s]
                        if (DEBUG):
                            print("NEW BEST r:", sentence)

            input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
            solutions = []
            evaluations = []
            for s in range(NEIGHBOR_SEARCH_NUM):
                # create neighbor sentence
                solution = self.generate(input_ids,greedy_probability=CONST.REC_GREEDY_PROB,MAX_TOKEN_GEN=CONST.REC_MAX_TOKEN_GEN)

                solutions.append(solution)
                evaluations.append(self.evaluate(solution, target_embed))

                # create neighbor sentence
                solution = self.rand_frag_add(input_ids,greedy_probability=CONST.REC_GREEDY_PROB,MAX_TOKEN_ADD=CONST.REC_MAX_TOKEN_ADD)

                solutions.append(solution)
                evaluations.append(self.evaluate(solution, target_embed))

            # choose best neighbor
            for s in range(len(solutions)):
                if sentence_eval < evaluations[s]:
                    sentence = solutions[s]
                    sentence_eval = evaluations[s]
                    if (DEBUG):
                        print("NEW BEST:", sentence)
        if (DEBUG):
            print("FINAL:", sentence)
        return sentence


def read_entry(index):
    """
        Pomocnicza funkcja do podglądu tweeta z OLID
        na podstawie indeksu.

        Parameters
        ----------
        index : int

        Returns
        -------
        str
            Treść tweeta.
    """
    base_path = "../../data/raw"
    train_path = os.path.join(base_path, "olid-training-v1.0.tsv")

    train_df = pd.read_csv(train_path, sep="\t")
    train_df = train_df[['tweet', 'subtask_a']].dropna()

    label_map = {"NOT": 0, "OFF": 1}
    labels = train_df['subtask_a'].map(label_map).tolist()
    texts = train_df['tweet'].tolist()

    print("Przykład tweeta:", texts[index])
    print("Etykieta:", labels[index])

    return texts[index]


# przyklad:
if __name__ == "__main__":
    toolkit = EmbedToolkit(INIT_GPT=True, INIT_ENCODER=True)
    embeddings = np.load("../../data/processed/embeddings.npy")
    print("shape", embeddings.shape)

    target = "I hate coffee breaks..."
    target_embed = toolkit.embedding(target)

    index = random.randint(0, embeddings.shape[0])
    print("INDEX:", index)
    target_embed = embeddings[index, :]
    read_entry(index)

    target_embed = torch.tensor(target_embed, dtype=torch.float32).to(toolkit.device)
    # target_embed.tolist()

    toolkit.search(target_embed=target_embed, EPOCHS=CONST.REC_EPOCHS, NEIGHBOR_SEARCH_NUM=CONST.REC_NEIGHBORS,
                   init_sentence="I love cats and I'm proud of it with all my heart!", DEBUG=True)
