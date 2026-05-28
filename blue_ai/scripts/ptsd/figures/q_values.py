from blue_ai.scripts.constants import DATA_PATH
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

names_map = {'HealthyAgent': 'healthy',
             'PTSDAgent': 'pre-existing\ndeficit',
             'TraumaSynapticDeficitAgent': 'post-trauma\ndeficit',
             }


for pane, file, title in [
    (1, 'q_values_forward_comparison_before_trauma.pkl', '"forward" q-values before trauma...'),
    (2, 'q_values_forward_comparison_after_trauma.pkl', '...immediately after trauma...'),
    (3, 'q_values_forward_comparison_after_relearning.pkl', '...and after relearning'),
]:

    plt.subplot(1, 3, pane)

    with open(DATA_PATH / 'results' / file, 'rb') as f:
        data = pickle.load(f)['results'].reset_index()
    data = data.melt(id_vars='agent', value_vars=['overall mean', 'trauma position '])
    data.columns = ['agent', 'variable', 'q value']
    data['variable'] = data['variable'].replace({'trauma position ': 'trauma state'})
    data['agent'] = data['agent'].replace(names_map)
    print(data)
    sns.barplot(data, x='agent', y='q value', hue='variable', palette=['grey', 'lightsteelblue'])
    plt.xlabel('')
    plt.legend().set_title("")
    plt.ylim([-1.2, 1.2])
    plt.title(title)

    if pane > 1:
        plt.gca().set_yticklabels(['' for _ in range(len(plt.gca().get_yticklabels()))])
        plt.ylabel('')
        plt.gca().legend_.remove()

plt.show()