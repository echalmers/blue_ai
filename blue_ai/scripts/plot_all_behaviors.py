import matplotlib.pyplot as plt

from view_performance import plot_learning_curves, plot_sample_env, plot_goals_per_episode
from view_discounting import plot_inferred_discount, plot_neuron_correlations
from view_perceived_fear_2 import plot_env_locations, plot_value_of_forward as plot_forward_value_fear
from view_perceived_value import plot_perceived_value_env, plot_perceived_value


mosaic = '''
a.b.c
d.e.c
'''

fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 9), width_ratios=[10, 1, 10, 1, 10])  #, height_ratios=[47.5, 5, 47.5])
plot_sample_env(axes['a'])
plot_goals_per_episode(axes['b'])
plot_inferred_discount(axes['d'])
plot_neuron_correlations(axes['e'])
plot_learning_curves(axes['c'])

# mosaic = '''
# e.f
# ...
# g.h
# '''
# fig2, axes2 = plt.subplot_mosaic(mosaic, figsize=(9, 9), width_ratios=[47.5, 5, 47.5], height_ratios=[47.5, 5, 47.5])
# plot_perceived_value_env(axes2['e'])
# plot_perceived_value(axes2['f'])
# plot_env_locations(axes2['g'])
# plot_forward_value_fear(axes2['h'])
# plt.plot([0, 1], [0.5, 0.5], color='grey', lw=1, transform=plt.gcf().transFigure, clip_on=False)

plt.show()
