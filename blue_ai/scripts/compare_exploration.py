from blue_ai.scripts.train_agents import load_trial
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.agents.agent_classes import (
    HealthyAgent,
    SpineLossDepression,
    ScaledTargets,
    ContextDependentLearningRate,
    HighDiscountRate,
    HighExploration,
)

import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import entropy

import os
from random import randint
import numpy as np
import pandas as pd
import pickle


all_agent_classes = [
    HealthyAgent,
    SpineLossDepression,
    ScaledTargets,
    ContextDependentLearningRate,
    HighDiscountRate,
    HighExploration,
]

SpineLossDepression.display_name = "simulated\nspine loss"
ScaledTargets.display_name = "reduced RPE"
ContextDependentLearningRate.display_name = "higher negative\nlearning rate"
HighDiscountRate.display_name = "high\ndiscounting"
HighExploration.display_name = "high\nexploration"


if os.path.exists(os.path.join(".", "data", "compare_exploration.pkl")):
    with open(os.path.join(".", "data", "compare_exploration.pkl"), "rb") as f:
        results = pickle.load(f)

else:

    def softmax(values, t=0.1):
        e_x = np.exp(
            values / t - np.max(values / t)
        )  # Subtracting the maximum value for numerical stability (thanks ChatGPT!)
        p = e_x / e_x.sum(axis=0)
        return p

    results = []

    for i in range(20):
        print(i)
        _, ref_agent, _ = load_trial(os.path.join(".", "data", f"HealthyAgent_{i}.pkl"))

        for j in range(4):
            for agent_class in all_agent_classes:

                if agent_class == HealthyAgent and i == j:
                    continue

                _, agent, _ = load_trial(
                    os.path.join(".", "data", f"{agent_class.__name__}_{j}.pkl")
                )

                env = Image2VecWrapper(
                    TransientGoals(
                        agent_start_pos=(randint(1, 5), randint(1, 5)),
                        agent_start_dir=randint(0, 3),
                        render_mode="rgb_array",
                    )
                )

                for rep in range(10):
                    state, _ = env.reset()
                    while state[1:, :, :].sum() == 0:
                        env = Image2VecWrapper(
                            TransientGoals(
                                agent_start_pos=(randint(1, 5), randint(1, 5)),
                                agent_start_dir=randint(0, 3),
                                render_mode="rgb_array",
                            )
                        )
                        state, _ = env.reset()

                    p1 = softmax(ref_agent.get_action_values(state).numpy())
                    p2 = softmax(agent.get_action_values(state).numpy())
                    results.append(
                        {
                            "agent": agent_class.display_name,
                            "entropy": entropy(p1, p2),
                            "dot": p1.dot(p2),
                            "max_agree": p1.argmax() == p2.argmax(),
                        }
                    )

    results = pd.DataFrame(results)
    with open(os.path.join(".", "data", "compare_exploration.pkl"), "wb") as f:
        pickle.dump(results, f)


def plot_kl_divergence(ax, **kwargs):
    plt.sca(ax)
    sns.barplot(
        data=results,
        x="agent",
        y="entropy",
        order=[agent.display_name for agent in all_agent_classes],
        **kwargs,
    )
    plt.ylabel("KL divergence from\nhealthy probability distribution over actions")
    plt.ylabel("")


if __name__ == "__main__":
    plt.figure()
    plot_kl_divergence(plt.gca())
    plt.show()
