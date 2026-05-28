from blue_ai.scripts.constants import DATA_PATH
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


names_map = {'Healthy': 'healthy',
             'PTSD': 'pre-existing\ndeficit',
             'PTSDAfterTrauma': 'post-trauma\ndeficit',
             }


plt.subplot(1, 2, 1)
with open(DATA_PATH / 'results' / 'object_recon_before_trauma.pkl', 'rb') as f:
    data = pickle.load(f)['results'].reset_index(names='agent')

data = data.melt(id_vars=['agent'], value_vars=['# Goals', '# Transient Goals', '# Hazards'])
data = data[data['agent'] != 'GroundTruth']
data['agent'] = data['agent'].replace(names_map)
data.columns = ['agent', 'variable', 'count']

sns.barplot(data, x='agent', hue='variable', y='count', palette=['tab:green', 'tab:blue', 'tab:red'])
plt.xlabel('')
plt.legend().set_title("")
plt.title('objects perceived by agents before trauma...')


plt.subplot(1, 2, 2)
with open(DATA_PATH / 'results' / 'object_recon_after_relearning.pkl', 'rb') as f:
    data = pickle.load(f)['results'].reset_index(names='agent')

data = data.melt(id_vars=['agent'], value_vars=['# Goals', '# Transient Goals', '# Hazards'])
data = data[data['agent'] != 'GroundTruth']
data['agent'] = data['agent'].replace(names_map)
data.columns = ['agent', 'variable', 'count']

sns.barplot(data, x='agent', hue='variable', y='count', palette=['tab:green', 'tab:blue', 'tab:red'])
plt.xlabel('')
plt.legend().set_title("")
plt.gca().legend_.remove()
plt.gca().set_yticklabels(['' for _ in range(len(plt.gca().get_yticklabels()))])
plt.ylabel('')
plt.title('...and after relearning')


plt.show()