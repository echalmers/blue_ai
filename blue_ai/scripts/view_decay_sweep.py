import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from blue_ai.scripts.constants import FIGURE_PATH, DATA_PATH

results = pd.read_csv(DATA_PATH / "decay_sweep.csv")
results["reward"] *= 100
print(results)
plt.figure(figsize=(8, 4))
sns.lineplot(results, x="decay", y="reward", palette=["grey"])
plt.xscale("log")
plt.xlabel("weight decay (λ)")
plt.ylabel("average reward per 100 steps")
plt.title("performance as weight decay increases")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "decay_sweep.png", dpi=300)
plt.show()
