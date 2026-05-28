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
    (2, 'q_values_forward_comparison_after_relearning.pkl', '...after relearning'),
    (3, 'q_values_forward_comparison_after_exposure_therapy_2.pkl', '...after "exposure therapy"...'),
    (4, 'q_values_forward_comparison_after_connectivity_restoration.pkl', '...and after relieving connectivity deficits'),
]:

    plt.subplot(1, 4, pane)

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
    plt.ylim([0, 1.2])
    plt.title(title)

    if pane > 1:
        plt.gca().set_yticklabels(['' for _ in range(len(plt.gca().get_yticklabels()))])
        plt.ylabel('')
        plt.gca().legend_.remove()

plt.show()