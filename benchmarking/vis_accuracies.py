import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    benchmarking_results = []

    for file_name in os.listdir("benchmarking/accuracies"):
        results = pd.read_csv(f"benchmarking/accuracies/{file_name}", index_col="T/T_m")
        benchmarking_results.append({"acc": results, "exp": file_name[: -4]})
    
    fig, axs = plt.subplots(2, (len(benchmarking_results) + 1) // 2, figsize=(10, 10))
    axs = axs.flatten()

    for i, info in enumerate(benchmarking_results):
        df = info["acc"]
        x = df.index

        for col in df.columns:
            y = df[col]
            axs[i].plot(x, y, label=col)
        
        axs[i].set_xlabel("T/T_m")
        axs[i].set_ylabel("Accuraacy")

        axs[i].legend()
        axs[i].set_title(info["exp"])

    plt.tight_layout()
    plt.show()
