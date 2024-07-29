from blue_ai.scripts.train_agents import load_trial
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from blue_ai.scripts.constants import DATA_PATH


weights = pd.DataFrame()

for file in [
    'HealthyAgent_0.pkl',
    'SpineLossDepression_0.pkl',
    'SchizophrenicAgent_0.pkl'
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

fig, ax = plt.subplots(1, 2)
plt.sca(ax[0])
sns.boxplot(data=weights[weights['type'] == 'weight'], x='agent', y='values')
plt.grid()
plt.title('connection weights')
plt.sca(ax[1])
sns.boxplot(data=weights[weights['type'] == 'bias'], x='agent', y='values')
plt.grid()
plt.title('neuron biases')
plt.show()