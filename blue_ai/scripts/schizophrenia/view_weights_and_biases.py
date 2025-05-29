from blue_ai.scripts.train_agents import load_trial
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from blue_ai.scripts.constants import DATA_PATH, _CURRENT_DIR


weights = pd.DataFrame()

for file in [
    'HealthyAgent_0.pkl',
    # 'SpineLossDepression_0.pkl',
    'SchizophrenicAgent_0.pkl',
    'ReverseImbalanceAgent_0.pkl'
]:
    _, agent, _ = load_trial(DATA_PATH / file)

    for paramset in agent.policy_net.parameters():
        weights = pd.concat((weights,
                             pd.DataFrame(
                                 {
                                     'agent': file,
                                     'type': 'weight' if paramset.dim() == 2 else 'bias',
                                     'values': list(paramset.cpu().detach().flatten().numpy())
                                 }
                             )
                             ))

weights['agent'] = weights['agent'].replace({'HealthyAgent_0.pkl': "healthy", 'SchizophrenicAgent_0.pkl': "schizophrenic"})

fig, ax = plt.subplots(1, 2)
plt.sca(ax[0])
p = sns.boxplot(data=weights[weights['type'] == 'weight'], x='agent', y='values', color=[0.5, 0.5, 0.5])
current_xlim = ax[0].get_xlim()
current_ylim = ax[0].get_ylim()
ax[0].fill_between(current_xlim, 0, current_ylim[1], color="green", alpha=0.1, linewidth=0)
ax[0].fill_between(current_xlim, current_ylim[0], 0, color="red", alpha=0.1, linewidth=0)
plt.text(x=0.25, y=0.75, s='excitatory', color='g')
plt.text(x=0.25, y=-0.75, s='inhibitory', color='r')
plt.xlim(current_xlim)
plt.ylim([-1, 1])
plt.grid()
plt.xlabel('')
plt.title('connection weights')
plt.ylabel('')

plt.sca(ax[1])
sns.boxplot(data=weights[weights['type'] == 'bias'], x='agent', y='values', color=[0.5, 0.5, 0.5])
plt.grid()
plt.xlabel('')
plt.title('neuron biases')
plt.ylabel('less excitability ←       → more excitability')
ax[1].yaxis.set_label_position('right')

plt.tight_layout()
plt.savefig(_CURRENT_DIR / 'schizophrenia' / 'img' / 'weights_biases.png', dpi=400)
plt.show()