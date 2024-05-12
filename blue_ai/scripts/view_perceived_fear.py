import pandas as pd

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.train_agents import load_trial
import matplotlib.pyplot as plt
import seaborn as sns

from constants import DATA_PATH, N_TRIALS

from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression


fig, ax = plt.subplots(1, 2)

env = Image2VecWrapper(
    TransientGoals(
        img_filename="env1.png",
        n_transient_goals=0,
        transient_obstacles=[(5, 5)],
        agent_start_pos=(5, 3),
        agent_start_dir=1,
        render_mode="rgb_array",
    )
)
state, _ = env.reset()
ax[0].imshow(env.render())
ax[0].set_xticks([])
ax[0].set_yticks([])

all_values = []

for trial in range(N_TRIALS):
    for dataset in [
        f"HealthyAgent_{trial}.pkl",
        f"SpineLossDepression_{trial}.pkl",
        f"ContextDependentLearningRate_{trial}.pkl",
    ]:
        results, agent, _ = load_trial(DATA_PATH / dataset)

        this_agent_values = agent.get_action_values(state).numpy()
        all_values.append([agent.display_name, "turn left", this_agent_values[0]])
        all_values.append([agent.display_name, "turn right", this_agent_values[1]])
        all_values.append([agent.display_name, "forward", this_agent_values[2]])

all_values = pd.DataFrame(data=all_values, columns=["agent", "action", "value"])
print(all_values)

plt.sca(ax[1])
p = sns.barplot(
    data=all_values,
    x="agent",
    y="value",
    hue="action",
    edgecolor=".5",
    hue_order=["turn left", "forward", "turn right"],
)
# p.patches[0].set_facecolor('skyblue')
# p.patches[2].set_facecolor('skyblue')
# p.patches[4].set_facecolor('skyblue')
# p.patches[1].set_facecolor('salmon')
# p.patches[3].set_facecolor('salmon')
# p.patches[5].set_facecolor('salmon')
# plt.legend([],[], frameon=False)
# plt.text(x=p.patches[0].get_center()[0], y=0.2, s='↶', fontsize='x-large', fontweight='heavy', ha='center')
# plt.text(x=p.patches[2].get_center()[0], y=0.2, s='↑', fontsize='x-large', fontweight='heavy', ha='center')
# plt.text(x=p.patches[4].get_center()[0], y=0.2, s='↷', fontsize='x-large', fontweight='heavy', ha='center')
# plt.text(x=p.patches[1].get_center()[0], y=0.2, s='↶', fontsize='x-large', fontweight='heavy', ha='center')
# plt.text(x=p.patches[3].get_center()[0], y=0.2, s='↑', fontsize='x-large', fontweight='heavy', ha='center')
# plt.text(x=p.patches[5].get_center()[0], y=0.2, s='↷', fontsize='x-large', fontweight='heavy', ha='center')

plt.xlabel("")
plt.ylabel("value perceived by agent")
# plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

fig.suptitle("anhedonia or fear generalization")
plt.show()
