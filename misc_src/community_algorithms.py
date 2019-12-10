## benchmark different community detection algorithms
from py3plex.algorithms.community_detection import community_wrapper as cw
from py3plex.algorithms.community_detection import community_measures as cm
from SCD import *
from sklearn.metrics.cluster import normalized_mutual_info_score,silhouette_score
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from py3plex.core import multinet
import argparse
import networkx.algorithms.community as commod
from collections import defaultdict
import signal
from contextlib import contextmanager


from py3plex.algorithms.community_detection import community_wrapper as cw
from py3plex.core import multinet
from py3plex.visualization.multilayer import *
from py3plex.visualization.colors import colors_default
from collections import Counter

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

def get_community_assignments(pairs,ground_truth):

    fpx = defaultdict(list)
    
    pairs_all = []
    for k,v in pairs.items():
        try:
            pairs_all.append((v,ground_truth[str(k)]))
            
        except Exception as es:
            print(es)
    
    for k,v in pairs.items():
        fpx[v].append(k)
        
    l1,l2 = zip(*pairs_all)
    return(l1,l2,fpx)


def visualize_a_partition(network,partition,method="test"):

    # select top n communities by size
    top_n = 1000
    partition_counts = dict(Counter(partition.values()))
    top_n_communities = list(partition_counts.keys())[0:top_n]

    # assign node colors
    color_mappings = dict(zip(top_n_communities,[x for x in colors_default if x != "black"][0:top_n]))

    try:
        network_colors = [color_mappings[partition[x]] if partition[x] in top_n_communities else "black" for x in network.nodes()]

        # visualize the network's communities!
        hairball_plot(network,
                      color_list=network_colors,
                      layout_parameters={"iterations": 100},
                      scale_by_size=True,
                      layout_algorithm="force",
                      legend=False)
        plt.savefig("./figures/{}".format(method),dpi=300)
        plt.clf()
    except:
        pass

def benchmark_on_dataset(graph,ground_truth,folder_path,network_tag,network_name):

    network = multinet.multi_layer_network()
    network.core_network = graph
    sparse_mat = nx.to_scipy_sparse_matrix(graph)
    node_set = network.core_network.nodes()
    NRC= SCD_obj(sparse_mat,node_names = node_set)
    num_important = 1000
    
    param1 = {"verbose":False,"sparisfy":False,"parallel_step":8,"prob_threshold":0.0005, "community_range":[5,sparse_mat.shape[1],10],"num_important":num_important,"clustering_measure":"silhouette","stopping":2,"improvement_step":0.005,"node_feature_type":"ppr_embedding"}

    param2 = {"verbose":False,"sparisfy":False,"parallel_step":8,"prob_threshold":0.0005, "community_range":[5,sparse_mat.shape[1],10],"num_important":num_important,"clustering_measure":"silhouette","stopping":5,"improvement_step":0.005,"node_feature_type":"netmf_embedding"}

    with timeout(10000):
        print("LP...")
        
       # print(list(commod.label_propagation.(network.core_network.to_undirected())))
        partition_lp = {enx:x for enx,x in enumerate(list(commod.label_propagation.asyn_lpa_communities(network.core_network)))}
        par_tmp = {}
        for k,v in partition_lp.items():
            for x in v:
                par_tmp[x] = int(k)
        partition_lp = par_tmp
    
    with timeout(10000):
        print("EBC...")
        partition_EBC= NRC.detect_communities(**param2)
    
    with timeout(10000):
        print("INM...")
        partition_infomap = cw.infomap_communities(network,binary="../bin/Infomap",multiplex=False,verbose=False)
    
    with timeout(3600):
        print("NORC...")
        partition_norc= NRC.detect_communities(**param1)  

    with timeout(10000):
        print("Louvain...")
        partition_louvain = cw.louvain_communities(network)        

    results = []
    ncl = cm.number_of_communities(partition_louvain)
    nci = cm.number_of_communities(partition_infomap)
    ncn = cm.number_of_communities(partition_norc)
    nce = cm.number_of_communities(partition_EBC)
    nclp = cm.number_of_communities(partition_lp)
    
    al, gl, com_l = get_community_assignments(partition_louvain,ground_truth)
    an, gn, com_norc = get_community_assignments(partition_norc,ground_truth)
    ai, gi, com_im = get_community_assignments(partition_infomap,ground_truth)
    aebc, gebc, com_EBC = get_community_assignments(partition_EBC,ground_truth)
    alp, glp, com_lp = get_community_assignments(partition_lp,ground_truth)

    if args.visualize_graphs == "True":
        partitions = [ground_truth,partition_EBC,partition_lp,partition_norc,partition_louvain,partition_infomap]
        ctx=0
        for partition in partitions:
            ctx+=1
            visualize_a_partition(graph,partition,ctx)
    
    NMI_louvain = normalized_mutual_info_score(al, gl)
    NMI_infomap = normalized_mutual_info_score(ai, gi)
    NMI_EBC = normalized_mutual_info_score(aebc, gebc)
    NMI_norc = normalized_mutual_info_score(an, gn)
    NMI_lp = normalized_mutual_info_score(alp, glp)

    ARI_louvain = adjusted_rand_score(al,gl)
    ARI_EBC = adjusted_rand_score(aebc,gebc)
    ARI_infomap = adjusted_rand_score(ai,gi)
    ARI_norc = adjusted_rand_score(an,gn)
    ARI_lp = adjusted_rand_score(alp,glp)
    
    louvain_modularity = cm.modularity(graph,com_l)
    infomap_modularity = cm.modularity(graph,com_im)
    norc_modularity = cm.modularity(graph,com_norc)
    EBC_modularity = cm.modularity(graph,com_EBC)
    lp_modularity = cm.modularity(graph,com_lp)

    out_object_EBC = {"Network_name": network_name, "folder_path": folder_path, "Network_tag":network_tag,"algorithm":"EBC","number_of_communities":nce,"modularity":EBC_modularity,"NMI":NMI_EBC,"ARI":ARI_EBC}
    
    out_object_lp = {"Network_name": network_name, "folder_path": folder_path, "Network_tag":network_tag,"algorithm":"LabelPropagation","number_of_communities":nclp,"modularity":lp_modularity,"NMI":NMI_lp,"ARI":ARI_lp}

    #out_object_async = {"Network_name": network_name, "folder_path": folder_path, "Network_tag":network_tag,"algorithm":"AsyncLP","number_of_communities":ncalp,"modularity":async_modularity,"NMI":NMI_async,"ARI":ARI_async}
    
    out_object_nc = {"Network_name": network_name, "folder_path": folder_path, "Network_tag":network_tag,"algorithm":"NoRC","number_of_communities":ncn,"modularity":norc_modularity,"NMI":NMI_norc,"ARI":ARI_norc}

    out_object_im = {"Network_name": network_name, "folder_path": folder_path, "Network_tag":network_tag,"algorithm":"Infomap","number_of_communities":nci,"modularity":infomap_modularity,"NMI":NMI_infomap,"ARI":ARI_infomap}

    out_object_lv = {"Network_name": network_name, "folder_path": folder_path, "Network_tag":network_tag,"algorithm":"Louvain","number_of_communities":ncl,"modularity":louvain_modularity,"NMI":NMI_louvain,"ARI":ARI_louvain}

  #  results.append(out_object_async)
    results.append(out_object_EBC)
    results.append(out_object_lp)
    results.append(out_object_nc)
    results.append(out_object_im)
    results.append(out_object_lv)
    
    for obj in results:
        print("\t".join(["RESULT_LINE",obj['Network_tag'],obj['algorithm'],str(obj['number_of_communities']),str(obj['modularity']),str(obj['NMI']),str(obj['ARI'])]))

    return results

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_network_folder",default="../data/parliament")
    parser.add_argument("--as_list", default="False")
    parser.add_argument("--network_tag", default="REAL_GRAPH")
    parser.add_argument("--visualize_graphs", default="True")

    args = parser.parse_args()
    network,labels = parse_network_to_object(args.input_network_folder,as_list=(args.as_list == "True"))

    network_name = args.input_network_folder.split("/")[-1]
    results = benchmark_on_dataset(network,labels,args.input_network_folder,args.network_tag,network_name)
    A = pd.DataFrame(results)
    import os
    directory = "results"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open('./results/results_{}_improved.tsv'.format(args.network_tag), 'a') as f:
        A.to_csv(f, header=False,sep="\t",columns=['Network_name','folder_path','Network_tag','algorithm','number_of_communities','modularity','NMI','ARI'])
