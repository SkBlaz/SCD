
from py3plex.visualization.multilayer import *
from py3plex.visualization.colors import colors_default
import SCD
from collections import Counter
import networkx as nx
from sklearn.decomposition import PCA
from scipy.sparse import csgraph

graph = nx.ring_of_cliques(30,3)
print(nx.info(graph))

sparse_mat = nx.to_scipy_sparse_matrix(graph)
node_names = list(range(sparse_mat.shape[1]))
SCD_detector = SCD.SCD_obj(sparse_mat,node_names=node_names) #instantiate

## all hyperparameters
SCD_detector.list_arguments()

## let's create a naive embedding with PCA on top of the laplacian matrix
laplacian = csgraph.laplacian(sparse_mat, normed=False)
normalized_pca = PCA(16).fit_transform(laplacian.todense())

## set hyperparameters
param2 = {"verbose":True,"parallel_step":8, "custom_embedding_vectors":normalized_pca}
partition = SCD_detector.detect_communities(**param2)

# select top n communities by size
top_n = 3 ## colors will repeat a bit
partition_counts = dict(Counter(partition.values()))
top_n_communities = list(partition_counts.keys())[0:top_n]

# assign node colors
color_mappings = dict(zip(top_n_communities,[x for x in colors_default if x != "black"][0:top_n]))
network_colors = [color_mappings[partition[x]] if partition[x] in top_n_communities else "black" for x in graph.nodes()]

# visualize the network's communities!
hairball_plot(graph,
	      color_list=network_colors,
              node_size = 100,
	      layout_parameters={"iterations": 20},
	      scale_by_size=False,
	      layout_algorithm="force",
	      legend=False)
plt.show()
