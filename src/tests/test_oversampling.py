from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

from src.config import CONSTANTS as CONST
from src.oversampling.LLMOversampler import LLMOversampler
from src.embeddings.EmbedToolkit import EmbedToolkit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def create_test_train_data(embed_toolkit):
    SEED = CONST.SEED
    data_path = CONST.PATH_DATASET_FULL

    train_df = pd.read_csv(data_path, sep="\t")
    train_df = train_df[['id','tweet', 'subtask_a']].dropna().reset_index(drop=True)

    #train_df['id'] = train_df.index

    label_map = {"NOT": 0, "OFF": 1}
    train_df['label'] = train_df['subtask_a'].map(label_map)

    # Teksty, etykiety, ID
    texts = train_df['tweet']
    labels = train_df['label']
    ids = train_df['id']

    # Podział na zbiór treningowy i testowy (80/20)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        texts,
        labels,
        ids,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    # Zamiana na listy
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()
    id_train = id_train.tolist()
    id_test = id_test.tolist()

    X_train_embeddings = []
    for text in X_train:
        emb = embed_toolkit.embedding(text)
        if hasattr(emb, 'cpu'):
            emb = emb.cpu().numpy()
        else:
            emb = np.array(emb)
        X_train_embeddings.append(emb)

    X_test_embeddings = []
    for text in X_test:
        emb = embed_toolkit.embedding(text)
        if hasattr(emb, 'cpu'):
            emb = emb.cpu().numpy()
        else:
            emb = np.array(emb)
        X_test_embeddings.append(emb)

    np.savez(
        CONST.PATH_TEST_OS_TRAINING_DATA,
        X=np.array(X_train, dtype=object),
        y=np.array(y_train),
        embeds=np.array(X_train_embeddings),
        id=np.array(id_train),
    )
    np.savez(
        CONST.PATH_TEST_OS_TEST_DATA,
        X=np.array(X_test, dtype=object),
        y=np.array(y_test),
        embeds=np.array(X_test_embeddings),
        id=np.array(id_test)
    )

def load_test_data():
    test_data = np.load(CONST.PATH_TEST_OS_TEST_DATA,allow_pickle=True)

    X = test_data['X'].tolist()
    y = test_data['y'].tolist()
    embeds = test_data['embeds'].tolist()
    id = test_data['id'].tolist()

    return X,y,embeds,id

def load_train_data():
    train_data = np.load(CONST.PATH_TEST_OS_TRAINING_DATA, allow_pickle=True)

    X = train_data['X'].tolist()
    y = train_data['y'].tolist()
    embeds = train_data['embeds'].tolist()
    id = train_data['id'].tolist()

    return X,y,embeds,id

def save_LLM_OS_data(samples,embeds,parent1_id,parent2_id):
    np.savez(
        CONST.PATH_TEST_OS_LLM,
        samples=np.array(samples, dtype=object),
        embeds=np.array(embeds),
        parent1_id=np.array(parent1_id),
        parent2_id=np.array(parent2_id)
    )
def load_LLM_OS_data():
    llm_data = np.load(CONST.PATH_TEST_OS_LLM, allow_pickle=True)

    samples = llm_data['samples'].tolist()
    embeds = llm_data['embeds'].tolist()
    parent1_id = llm_data['parent1_id'].tolist()
    parent2_id = llm_data['parent2_id'].tolist()
    return samples,embeds,parent1_id,parent2_id

def concat_LLM_OS_data():
    llm_data = np.load(CONST.PATH_TEST_DATA_OVERSAMPLING + "os_LLM_1.npz", allow_pickle=True)
    llm_data2 = np.load(CONST.PATH_TEST_DATA_OVERSAMPLING + "os_LLM_2.npz", allow_pickle=True)
    llm_data3 = np.load(CONST.PATH_TEST_DATA_OVERSAMPLING + "os_LLM_3.npz", allow_pickle=True)

    samples = llm_data['samples'].tolist()
    samples += llm_data2['samples'].tolist()
    samples += llm_data3['samples'].tolist()
    embeds = llm_data['embeds'].tolist()
    embeds += llm_data2['embeds'].tolist()
    embeds += llm_data3['embeds'].tolist()
    parent1_id = llm_data['parent1_id'].tolist()
    parent1_id += llm_data2['parent1_id'].tolist()
    parent1_id += llm_data3['parent1_id'].tolist()
    parent2_id = llm_data['parent2_id'].tolist()
    parent2_id += llm_data2['parent2_id'].tolist()
    parent2_id += llm_data3['parent2_id'].tolist()

    np.savez(
        CONST.PATH_TEST_OS_LLM,
        samples=np.array(samples, dtype=object),
        embeds=np.array(embeds),
        parent1_id=np.array(parent1_id),
        parent2_id=np.array(parent2_id)
    )

def LLM_main():
    embed_toolkit = EmbedToolkit(INIT_GPT=False,INIT_ENCODER=True)
    #create_test_train_data(embed_toolkit)
    #print("data created")
    llm_oversampler = LLMOversampler(embed_toolkit)
    X,y,embeds,id_list = load_train_data()
    X_minority = [x for x, label in zip(X, y) if label == 1]
    id_minority = [i for i, label in zip(id_list, y) if label == 1]
    embeds_minority = [embed for embed, label in zip(embeds, y) if label == 1]
    samples,over_embeds,parent1_id,parent2_id= llm_oversampler.oversample_data(X_minority,id_minority,embeds_minority)
    save_LLM_OS_data(samples,over_embeds,parent1_id,parent2_id)
    samples,over_embeds,parent1_id,parent2_id = load_LLM_OS_data()

def save_SMOTE_OS_data(samples, embeds, parent1_id, parent2_id):
    np.savez(
        CONST.PATH_TEST_OS_SMOTE,
        samples=np.array(samples, dtype=object),
        embeds=np.array(embeds),
        parent1_id=np.array(parent1_id),
        parent2_id=np.array(parent2_id),
    )

def load_SMOTE_OS_data():
    data = np.load(CONST.PATH_TEST_OS_SMOTE, allow_pickle=True)

    return (
        data["samples"].tolist(),
        data["embeds"].tolist(),
        data["parent1_id"].tolist(),
        data["parent2_id"].tolist(),
    )

from src.oversampling.SMOTEOversampler import SMOTEOversampler

def SMOTE_main():
    X, y, embeds, id_list = load_train_data()

    X_minority = [x for x, label in zip(X, y) if label == 1]
    id_minority = [i for i, label in zip(id_list, y) if label == 1]
    embeds_minority = [e for e, label in zip(embeds, y) if label == 1]

    smote = SMOTEOversampler(k=5, seed=CONST.SEED)

    samples, over_embeds, parent1_id, parent2_id = smote.oversample_data(
        X_minority,
        id_minority,
        embeds_minority,
        n_samples=len(embeds_minority)
    )

    save_SMOTE_OS_data(samples, over_embeds, parent1_id, parent2_id)

    print("SMOTE oversampling finished")

def evaluate_1nn(X_train_emb, y_train, X_test_emb, y_test):
    """
    Ewaluacja 1-NN, metryki tylko dla klasy OFF (1)
    """
    clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[1],        # tylko OFF
        average="binary",
        zero_division=0
    )

    return precision, recall, f1

def build_original_train():
    _, y, embeds, _ = load_train_data()
    return np.array(embeds), np.array(y)


def build_smote_train():
    _, y, embeds, _ = load_train_data()
    _, smote_embeds, _, _ = load_SMOTE_OS_data()

    X_train = np.vstack([embeds, smote_embeds])
    y_train = np.concatenate([y, np.ones(len(smote_embeds))])

    return X_train, y_train


def build_llm_train():
    _, y, embeds, _ = load_train_data()
    _, llm_embeds, _, _ = load_LLM_OS_data()

    X_train = np.vstack([embeds, llm_embeds])
    y_train = np.concatenate([y, np.ones(len(llm_embeds))])

    return X_train, y_train

def run_knn_experiment():
    _, y_test, test_embeds, _ = load_test_data()
    X_test_emb = np.array(test_embeds)
    y_test = np.array(y_test)

    results = []

    for name, builder in [
        ("ORIGINAL", build_original_train),
        ("SMOTE", build_smote_train),
        ("LLM", build_llm_train),
    ]:
        X_train_emb, y_train = builder()

        precision, recall, f1 = evaluate_1nn(
            X_train_emb,
            y_train,
            X_test_emb,
            y_test
        )

        results.append({
            "dataset": name,
            "precision_OFF": precision,
            "recall_OFF": recall,
            "f1_OFF": f1
        })

        print(f"{name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

    return pd.DataFrame(results)

def save_knn_results(df):
    path = CONST.PATH_TEST_DATA + "knn_1_results.csv"
    df.to_csv(path, index=False)
    print(f"Wyniki zapisane do: {path}")



if __name__ == "__main__":
    # 1) jeśli już masz training_data.npz → NIE uruchamiaj ponownie
    # embed_toolkit = EmbedToolkit(INIT_GPT=False, INIT_ENCODER=True)
    # create_test_train_data(embed_toolkit)

    # 2) oversampling (jeśli już masz pliki → można zakomentować)
    # SMOTE_main()
    # LLM_main()

    # 3) EKSPERYMENT 1-NN
    df = run_knn_experiment()
    save_knn_results(df)
