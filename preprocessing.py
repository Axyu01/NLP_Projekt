import pandas as pd
import os

base_path = "dataset"

train_path = os.path.join(base_path, "olid-training-v1.0.tsv")

train_df = pd.read_csv(train_path, sep="\t")

train_df.head()

train_df['subtask_a'].value_counts()
class_counts = train_df['subtask_a'].value_counts()
class_percent = class_counts / class_counts.sum() * 100

print("Class counts:\n", class_counts)
print("\nClass percentage:\n", class_percent)

train_df = train_df[['tweet', 'subtask_a']].dropna()

label_map = {"NOT": 0, "OFF": 1}
labels = train_df['subtask_a'].map(label_map).tolist()

texts = train_df['tweet'].tolist()

print(texts[0], labels[0])