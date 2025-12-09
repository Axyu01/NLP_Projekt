import evaluation_env as ev
import matplotlib.pyplot as plt

texts = [
    "I hate you so much",
    "You are a terrible person",
    "I really love you and appreciate your help",
    "I want to punch you in the face",
    "Have a nice day, you are amazing",
]

target = "I hate you so much"

df = ev.evaluate_text_list(texts, k=1, target_text=target)
print(df)

print("\nPodsumowanie etykiet (pred_label_name):")
print(df["pred_label_name"].value_counts())

if "sim_to_target" in df.columns:
    print("\nStatystyki sim_to_target:")
    print(df["sim_to_target"].describe())


# if "sim_to_target" in df.columns:
#     plt.figure()
#     plt.hist(df["sim_to_target"], bins=10)
#     plt.title("Histogram sim_to_target dla przykładowych tekstów")
#     plt.xlabel("sim_to_target")
#     plt.ylabel("Liczba tekstów")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("sim_to_target_example.png", dpi=200)
#     print("\nZapisano wykres: sim_to_target_example.png")
