from blue_ai.scripts.train_dqn import TrialRunner, load_trial, Image2VecWrapper, StaticDropout
import torch
import os
import matplotlib.pyplot as plt

dropout = []

def set_dropout(**kwargs):
    current_p = None
    for net in [kwargs['agent'].value_net, kwargs['agent'].policy_net]:
        for layer in net:
            if isinstance(layer, torch.nn.modules.dropout.Dropout) or isinstance(layer, StaticDropout):
                if kwargs['step'] == 15000:
                    layer.p = 0.5
                elif kwargs['step'] == 25000:
                    layer.p = 0
                current_p = layer.p
    dropout.append(current_p)


for trial in range(10):
    results, agent = TrialRunner(
        dropout=0,
        filename=os.path.join('.', 'data', f'rehabilitate_{trial}.pkl'),
        trial_id=1,
        allow_done_action=False,
        callbacks=[set_dropout],
        steps=40000
    )()

plt.figure()
ax = plt.gca()
ax.plot(results['cumulative_reward'])

plt.show()