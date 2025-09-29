from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.ptsd.train_agents import load_results, save_results

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import sys

def view_reconstruction_change (directory: Path, show_plot: bool = True):
    """
    Compare agents' reconstructed hazards before trauma and after relearning.

    This function loads reconstruction results saved as `.pkl` files, computes the 
    proportion of hazards relative to all reconstructed elements (goals, transient goals, 
    and hazards), and compares the values before trauma and after relearning. Results 
    are combined into a summary DataFrame, saved to disk, and optionally plotted.

    Args:
        directory (Path): Path to the directory containing the agents’ reconstruction results.
        show_plot (bool, optional): Whether to display the plot after saving it. Defaults to True.

    """

    files = {
        "before": "object_recon_before_trauma.pkl",
        "after": "object_recon_after_relearning.pkl",
    }
    
    def compute_proportion(df, label):
        df[f"{label}"] = df["# Hazards"] / (
            df["# Goals"] + df["# Transient Goals"] + df["# Hazards"] + 1e-9
        )
        return df

    # load and compute
    results = {}
    for label, file in files.items():
        df = load_results(f"{directory}/results/{file}")
        results[label] = compute_proportion(df, label)
    
    # combine into a single DataFrame
    final = results["before"][["before"]].copy()
    final["after"] = results["after"]["after"]
    final = final.reset_index().rename(columns={"index": "Agent"})
    final = final[final["Agent"] != "GroundTruth"]
    print(final)

    # convert in long for seaborn
    final_long = final.melt(id_vars="Agent", value_vars=["before", "after"],
                  var_name="time", value_name="Proportion")
    print(final_long)

    # plot and save
    img_path = directory / "img"
    img_path.mkdir(parents=True, exist_ok=True)

    results_path = directory / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    subpath = "change_in_reconstructed_hazards.png"

    save_results(final, results_path / subpath.replace("png", "pkl"))

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=final_long, x="time", y="Proportion", hue="Agent", marker="o")
    plt.title("The proportion of reconstructed hazards before trauma vs. after relearning")
    plt.savefig(img_path / subpath)
    if show_plot:
        plt.show()


if __name__ == "__main__":
    view_reconstruction_change(DATA_PATH / sys.argv[1])