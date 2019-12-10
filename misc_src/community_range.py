## benchmark different community detection algorithms
from norc import *
import json
from sklearn.metrics.cluster import normalized_mutual_info_score,silhouette_score
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from py3plex.core import multinet
import argparse
import networkx.algorithms.community as commod

import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError

def get_community_assignments(partition,ground_truth):

    pairs = {}
    for k,v in partition.items():
        for el in v:
            pairs[el] = k

    pairs_all = []
    for k,v in pairs.items():
        try:
            pairs_all.append((v,ground_truth[k]))
        except:
            pass

    l1,l2 = zip(*pairs_all)
    return(l1,l2)

def benchmark_on_dataset(graph,ground_truth,folder_path,network_tag,network_name):

    network = multinet.multi_layer_network()
    network.core_network = graph
    sparse_mat = nx.to_scipy_sparse_matrix(graph)
    NRC= NoRC(sparse_mat)
    node_set = network.core_network.nodes()
    num_important = 1000
    
    param2 = {"verbose":False,"sparisfy":False,"parallel_step":8,"prob_threshold":0.0005, "community_range":[5,sparse_mat.shape[1],10],"num_important":num_important,"node_names":node_set,"clustering_measure":"silhouette","stopping":3,"improvement_step":0.005,"node_feature_type":"embedding"}

    print("EBC...")
    partition_EBC= NRC.detect_communities(**param2)
    try:
        cluster_quality = NRC.cluster_quality
        with open('traces/traces_{}.json'.format(network_name), 'w') as outfile:
            json.dump(cluster_quality, outfile)
    except Exception as es:
        print(es)
        pass

def parse_network_to_object(folder,directed=False,as_list=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    with open(folder+"/network.dat") as net:
        for line in net:
            if line.strip()[0] != "#":
                n1,n2= line.strip().split()
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
                    node_community_map[n1] = n2

    return (G,node_community_map)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_network_folder",default="./LFR/initial")
    parser.add_argument("--as_list", default="False")
    parser.add_argument("--network_tag", default="REAL_GRAPH")

    args = parser.parse_args()
    import os
    directory = "traces"
    if not os.path.exists(directory):
        os.makedirs(directory)
    network,labels = parse_network_to_object(args.input_network_folder,as_list=(args.as_list == "True"))
    network_name = args.input_network_folder.split("/")[-1]
    results = benchmark_on_dataset(network,labels,args.input_network_folder,args.network_tag,network_name)
    A = pd.DataFrame(results)
