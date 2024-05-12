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

from random import randint
import pandas as pd
import pickle

from constants import DATA_PATH, N_TRIALS


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


if (DATA_PATH / "compare_exploration.pkl").exists():
    with open(DATA_PATH / "compare_exploration.pkl", "rb") as f:
        results = pickle.load(f)

else:

    from blue_ai.agents.dqn import softmax

    results = []

    for i in range(N_TRIALS):
        _, ref_agent, _ = load_trial(DATA_PATH / f"HealthyAgent_{i}.pkl")

        # this was previously set too 4, probably for performance reasons
        for j in range(N_TRIALS):
            for agent_class in all_agent_classes:

                # We don't want to compare the agent to itself
                if agent_class == HealthyAgent and i == j:
                    continue

                _, agent, _ = load_trial(DATA_PATH / f"{agent_class.__name__}_{j}.pkl")

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

                    p1 = softmax(ref_agent.get_action_values(state), t=0.1)
                    p2 = softmax(agent.get_action_values(state), t=0.1)

                    results.append(
                        {
                            "agent": agent_class.display_name,
                            "entropy": entropy(p1.cpu(), p2.cpu()),
                            "dot": p1.dot(p2),
                            "max_agree": p1.argmax() == p2.argmax(),
                        }
                    )

    results = pd.DataFrame(results)
    with open(DATA_PATH / "compare_exploration.pkl", "wb") as f:
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
