from py3plex.visualization.multilayer import *
from py3plex.visualization.colors import colors_default
import SCD
from collections import Counter


import networkx as nx

graph = nx.caveman_graph(20, 10)
print(nx.info(graph))

sparse_mat = nx.to_scipy_sparse_matrix(graph)
node_names = list(range(sparse_mat.shape[1]))
SCD_detector = SCD.SCD_obj(sparse_mat,node_names=node_names) #instantiate

## all hyperparameters
SCD_detector.list_arguments()

## set hyperparameters
param2 = {"verbose":True,"parallel_step":8}
partition = SCD_detector.detect_communities(**param2)

# select top n communities by size
top_n = 20 ## colors will repeat a bit
partition_counts = dict(Counter(partition.values()))
top_n_communities = list(partition_counts.keys())[0:top_n]

# assign node colors
color_mappings = dict(zip(top_n_communities,[x for x in colors_default if x != "black"][0:top_n]))
network_colors = [color_mappings[partition[x]] if partition[x] in top_n_communities else "black" for x in graph.nodes()]

# visualize the network's communities!
hairball_plot(graph,
	      color_list=network_colors,
	      layout_parameters={"iterations": 20},
	      scale_by_size=True,
	      layout_algorithm="force",
	      legend=False)
plt.show()
