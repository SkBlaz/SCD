## parse affinomics to a simple network.
## detect communities.
import networkx as nx
from collections import defaultdict
from SCD import *
from py3plex.visualization.multilayer import *
from py3plex.visualization.colors import colors_default
from collections import Counter

from py3plex.core import multinet
from py3plex.algorithms import hedwig
from py3plex.algorithms.hedwig.core.converters import *
from py3plex.algorithms.community_detection import community_wrapper as cw
from py3plex.algorithms.statistics import enrichment_modules

import rdflib

def communities_to_file(partition):

    partitions = defaultdict(list)
    for k,v in partition.items():
        partitions[v].append(k)

    fx = open("semantic_outputs/partitions.tsv","w")
    for k,v in partitions.items():
        vx = "\t".join(v)
        fs=str(k)+"\t"+vx+"\n"
        fx.write(fs)
    fx.close()

def visualize_a_partition(network,partition,method="test"):

    # select top n communities by size
    top_n = 50
    partition_counts = dict(Counter(partition.values()))
    top_n_communities = list(partition_counts.keys())[0:top_n]

    # assign node colors
    color_mappings = dict(zip(top_n_communities,[x for x in colors_default if x != "black"][0:top_n]))

    try:
        network_colors = [color_mappings[partition[x]] if partition[x] in top_n_communities else "black" for x in network.nodes()]

        # visualize the network's communities!
        hairball_plot(network,
                      color_list = network_colors,
                      node_size = 30,
                      edge_width = 0.1,
                      layout_parameters = {"iterations": 100},
                      scale_by_size=False,
                      layout_algorithm = "force",
                      legend=False)
        plt.show()
        plt.clf()
    except Exception as es:
        print(es)

def read_go_mapping_file(gaf):
    
    maps = defaultdict(list)
    with open(gaf) as gfile:
        for line in gfile:
            line = line.strip().split("\t")
            if line[0] == "UniProtKB":
                maps[line[1]].append(line[4])
    return maps

def parse_PSI_to_edgelist(psi_file):

    G = nx.Graph()
    with open(psi_file) as pf:        
        for line in pf:
            line = line.strip()
            parts = line.split("\t")
            node_first = parts[0]
            node_second = parts[1]
            if node_first.count(" ") < 1 and "#ID" not in node_first and node_second.count(" ")<1:
                G.add_edge(node_first,node_second)
            else:
                print(node_first)
    return G

if __name__ == "__main__":

    psi_file = "bio_example/affinomics.tsv"
    gaf_file = "bio_example/goa_human.gaf"

    annotations = read_go_mapping_file(gaf_file)
    graph = parse_PSI_to_edgelist(psi_file)
    
    print(nx.info(graph))
    global_params= {"verbose":True,"parallel_step":8,
                    "community_range":[5,len(list(graph.nodes())),5],
                    "clustering_measure":"silhouette",
                    "stopping":5,
                    "output_format" : "paired",
                    "improvement_step":0.005,
                    "node_feature_type":"netmf_embedding"}
    embedding_space = {"negative_range" : [1],
                       "window_sizes" : [4,5],
                       "dims" : [128]}
    
    SCD = SCD_obj(graph, input_type="networkx")
    partitions= SCD.detect_communities(**global_params,**embedding_space)
    communities_to_file(partitions)
    visualize_a_partition(graph, partitions)
    ## semantic induction comes here.
    tmp_gaf = "./BK/goa_human.gaf"
    rdf_partitions = hedwig.convert_mapping_to_rdf(partitions,annotation_mapping_file=tmp_gaf,layer_type="uniprotkb",go_identifier=None,prepend_string="GO:")
    rdf_partitions.serialize(destination = "semantic_inputs/semantic_annotations.n3", format="n3")
    obo2n3("BK/go-basic.obo","BK/ontology.n3", "./BK/goa_human.gaf")
    
    hedwig_input_parameters = {"bk_dir": "BK/",
                               "data": "semantic_inputs/semantic_annotations.n3",
                               "format": "n3",
                               "output": "semantic_outputs/rules.json",
                               "covered": None,
                               "mode": "subgroups",
                               "target": None,
                               "score": "lift",
                               "negations": True,
                               "alpha": 0.05,
                               "latex_report": False,
                               "adjust": "fwer",
                               "FDR": 0.1,
                               "leaves": True,
                               "learner": "heuristic",
                               "optimalsubclass": False,
                               "uris": False,
                               "beam": 30,
                               "support": 0.01,
                               "depth": 10,
                               "nocache": True,
                               "verbose": False,
                               "adjust":"fdr"}
    
    hedwig_results = hedwig.run(hedwig_input_parameters)
