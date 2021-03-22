## a simple test
from collections import defaultdict
import networkx as nx

from sklearn.metrics.cluster import normalized_mutual_info_score,silhouette_score
import networkx.algorithms.community as commod


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

def get_community_assignments(pairs,ground_truth):

    fpx = defaultdict(list)
    
    # For each node, store real and assumed community ID
    pairs_all = []
    for k,v in pairs.items():
        try:
            pairs_all.append((v,ground_truth[str(k)]))
        except Exception as es:
            print(es)
    for k,v in pairs.items():
        fpx[v].append(k)        
    l1,l2 = zip(*pairs_all)
    assert len(l1) == len(l2)
    return(l1,l2,fpx)

fname = "./5000__15__50__2__1__0.7"
G, ground_truth = parse_network_to_object(fname)
partition_lp = {enx:x for enx,x in enumerate(list(commod.label_propagation.asyn_lpa_communities(G)))}

par_tmp = {}
for k,v in partition_lp.items():
    for x in v:
        par_tmp[x] = int(k)
        
partition_lp = par_tmp
alp, glp, com_lp = get_community_assignments(partition_lp,ground_truth)

print(len(alp), len(glp))
NMI_lp = normalized_mutual_info_score(alp, glp, average_method = "geometric")
print(NMI_lp)
