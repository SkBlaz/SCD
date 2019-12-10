## plot optimization trace

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style("whitegrid")
dfx = pd.read_csv("optimization_trace.tsv",sep="\t")
#sns.lineplot(dfx.dimension,dfx.score,hue=dfx.window)
sns.violinplot(dfx.dimension, dfx.score)
plt.ylabel("Silhouette score")
plt.xlabel("Embedding dimension")
plt.show()
