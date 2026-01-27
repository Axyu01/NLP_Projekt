import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV = "../../test_data/knn_1_results.csv"

OUT_TABLE_CSV = "pytanie3_knn_results_table.csv"
OUT_COMBINED_PLOT = "pytanie3_precision_recall_f1_barplot.png"

df = pd.read_csv(INPUT_CSV)

print("\n=== WYNIKI 1-NN (OFF) ===")
print(df.to_string(index=False))

df.to_csv(OUT_TABLE_CSV, index=False)
print(f"\nTabela zapisana do: {OUT_TABLE_CSV}")

x = np.arange(len(df))
width = 0.25

plt.figure(figsize=(8, 4))

plt.bar(x - width, df["precision_OFF"], width, label="Precision")
plt.bar(x, df["recall_OFF"], width, label="Recall")
plt.bar(x + width, df["f1_OFF"], width, label="F1-score")

plt.xticks(x, df["dataset"])
plt.ylabel("Score")
plt.xlabel("Dataset")
plt.title("Precision, Recall i F1-score (1-NN, klasa OFF)")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(OUT_COMBINED_PLOT, dpi=200)
plt.close()

print(f"Wykres Precision/Recall/F1 zapisany do: {OUT_COMBINED_PLOT}")
