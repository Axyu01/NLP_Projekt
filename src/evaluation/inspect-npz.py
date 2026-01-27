import numpy as np

FILES = [
    '/Users/nataliamaciewicz/Documents/studia/SM/SEM 2/NLP/NLP_Projekt/test_data/oversampling/os_LLM_1.npz',
    '/Users/nataliamaciewicz/Documents/studia/SM/SEM 2/NLP/NLP_Projekt/test_data/oversampling/os_LLM_2.npz',
    '/Users/nataliamaciewicz/Documents/studia/SM/SEM 2/NLP/NLP_Projekt/test_data/oversampling/os_LLM_3.npz',
    '/Users/nataliamaciewicz/Documents/studia/SM/SEM 2/NLP/NLP_Projekt/test_data/oversampling/os_LLM.npz',
    # "test_data/oversampling/os_LLM_2.npz",
    # "test_data/oversampling/os_LLM_3.npz",
    # "test_data/oversampling/os_LLM.npz",
]

for path in FILES:
    print("=" * 60)
    print("FILE:", path)
    data = np.load(path, allow_pickle=True)

    print("Keys:", list(data.keys()))

    for k in data.keys():
        v = data[k]
        try:
            print(f"  {k}: type={type(v)}, shape={v.shape}, len={len(v)}")
        except Exception:
            print(f"  {k}: type={type(v)}")

    if "samples" in data:
        print("\nExample sample:")
        print(data["samples"][0])
