import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_scores(hists, labels, title='scores'):
  fig, ax = plt.subplots(figsize=(10, 6), dpi=130)

  bot, top = 1, 0
  for i, h in enumerate(hists):
    ax.plot(h, label=labels[i])
    ax.scatter(range(len(h)), h, s=13, )
    if np.max(h) > top:
      top = 1.05 * np.max(h)

    if np.min(h) < bot:
      bot = 0.95 * np.min(h)

  ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
  ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
  ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

  ax.set_ylim(bottom=bot, top=top)
  ax.grid(True)
  ax.legend()

  ax.set_title('Incremental classification accuracy')
  ax.set_xlabel('Query iteration')
  ax.set_ylabel('Classification Accuracy')

  plt.savefig(title)

      
def get_initial_indexes(y, n_items=50):
  cl = np.unique(y)
  indexes = []
  limit = int(n_items / len(cl))
  for c in cl:
    values = [int(i) for i,v in enumerate(y)  if v == c]
    indexes = indexes + values[:limit]

  indexes = sorted(indexes)
  return indexes