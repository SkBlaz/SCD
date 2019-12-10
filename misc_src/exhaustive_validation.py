## exhaustive validation..
from SCD import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import json
from collections import defaultdict
sns.set_style("whitegrid")
def parse_network_to_object(folder,directed=False,as_list=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    with open(folder+"/network.dat") as net:
        for line in net:
            if line.strip()[0] != "#":
                n1,n2= line.strip().split()
                if n1 != n2:
                    G.add_edge(n1,n2)

    node_community_map = {}
    with open(folder+"/community.dat") as net:
        community_counter = 0
        for line in net:
            if line.strip()[0] != "#":
                if as_list:
                    nodes = line.strip().split()
                    for node in nodes:
                        node_community_map[node] = community_counter
                    community_counter += 1
                else:
                    n1,n2= line.strip().split()
                    node_community_map[n1] = int(n2)

    return (G,node_community_map)

network_folders = ["../data/parliament"]
out_struct = {}

for input_network_folder in network_folders:
    graph,labels = parse_network_to_object(input_network_folder,as_list=True)
    sparse_mat = nx.to_scipy_sparse_matrix(graph)
    krange = [1,5,10,20,50,int(np.power(100,2/3)),int(np.power(300,2/3))]
    for j in krange: ## range of potential steps
        node_set = graph.nodes()
        SCD = SCD_obj(sparse_mat,node_names = node_set)
        if j == 1:
            stop = 500
        else:
            stop = 5
             
        param2 = {"verbose":False,"sparisfy":False,"parallel_step":8,"prob_threshold":0.0005, "community_range":[10,300,j],"clustering_measure":"silhouette","stopping":stop,"improvement_step":0.005,"node_feature_type":"netmf_embedding", "use_normalized_scores":False}
        partition_EBC= SCD.detect_communities(**param2)
        out_struct[j] = SCD.all_scores

with open('exhaustive.pickle', 'wb') as handle:
    pickle.dump(out_struct, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
for j in krange:
    data = out_struct[j]
    x = [x[0] for x in data]
    y = [x[1] for x in data]
    max_im = np.argmax(y)
    max_point_x = x[max_im]
    max_point_y = y[max_im]
    plt.plot(max_point_x,max_point_y,label="max; k = {}".format(j),marker="v")
    plt.scatter(x,y,label="k = {}".format(j),s=12)
plt.legend()
plt.xlabel("Number of communities (k)")
plt.ylabel("Silhouette score")
plt.tight_layout()
plt.savefig("figures/brute.pdf",dpi=300)
