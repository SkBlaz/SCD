import SCD
import scipy.io
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx

#graph = nx.caveman_graph(20, 10)
#print(nx.info(graph))

graph = scipy.io.loadmat("../example_networks/example_network.mat")
sparse_mat = graph['network']
#sparse_mat = nx.to_scipy_sparse_matrix(graph)
node_names = list(range(sparse_mat.shape[1]))
SCD_detector = SCD.SCD_obj(sparse_mat,node_names=node_names) #instantiate

## all hyperparameters
SCD_detector.list_arguments()

## set hyperparameters
param2 = {"verbose":True,"parallel_step":8, "community_range":[20,100,10]}
partition = SCD_detector.detect_communities(**param2)

graph = nx.from_scipy_sparse_matrix(sparse_mat)
values = [partition[x] for x in node_names]
pos = nx.spring_layout(graph, iterations = 300)
nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), 
                       node_color = values, node_size = 10)
nx.draw_networkx_edges(graph, pos)
plt.show()
