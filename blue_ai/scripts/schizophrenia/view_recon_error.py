from blue_ai.scripts.constants import _CURRENT_DIR, DATA_PATH
import matplotlib.transforms as mtransforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mosaic = """
    a
    b
    """
fig, axes = plt.subplot_mosaic(
    mosaic,
    figsize=(11, 8),
    # width_ratios=[10, 10],
    # height_ratios=[47.5, 5, 47.5],
)


datapoints = pd.read_csv(_CURRENT_DIR / 'schizophrenia' / 'img' / 'recon_error_data.csv')
datapoints['agent'] = datapoints['agent'].str.lower()
plt.sca(axes['a'])
sns.lineplot(data=datapoints[datapoints['std'] <= 0.3], x='std', y='mse', hue='agent', palette=['blue', 'orange'])
plt.ylabel('mean-squared error of reconstruction')
plt.xlabel('standard deviation of added noise')
plt.title('error in reconstructing visual input')


for fname, ax in (('1741297681.590705.png', axes['b']),):  #('1741297930.701264.png', axes['b']),)):
    im = plt.imread(DATA_PATH / 'hallucinations' / fname)
    ax.imshow(im[800:2100, 680:5200, :])
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])


for label, ax in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize="large",
        weight="bold",
        va="bottom",
        fontfamily="serif",
    )

plt.savefig(_CURRENT_DIR / 'schizophrenia' / 'img' / 'recon_error.png', dpi=400)
plt.tight_layout()
plt.show()