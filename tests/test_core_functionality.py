import scipy.io
import SCD
import networkx as nx

def test_synthetic_cd():

    graph, cmap = SCD.parse_network_to_object("example_networks/parliament",directed=True,as_list=False)
    sparse_mat = nx.to_scipy_sparse_matrix(graph)
    node_names = list(range(sparse_mat.shape[1]))
    SCD_detector = SCD.SCD_obj(sparse_mat,node_names=node_names) #instantiate

    ## all hyperparameters
    SCD_detector.list_arguments()

    ## set hyperparameters
    param2 = {"verbose":True,"parallel_step":8}
    communities = SCD_detector.detect_communities(**param2)
    print(communities)


def test_real_cd():

    ## load a sparse matrix representation of the network
    graph = scipy.io.loadmat("example_networks/example_network.mat")
    graph = graph['network']
    node_names = list(range(graph.shape[1]))
    SCD_detector = SCD.SCD_obj(graph,node_names=node_names) #instantiate

    ## all hyperparameters
    SCD_detector.list_arguments()

    ## set hyperparameters
    param2 = {"verbose":True,"parallel_step":8}
    communities = SCD_detector.detect_communities(**param2)
    print(communities)
