import SCD
import scipy.io
## load a sparse matrix representation of the network
graph = scipy.io.loadmat("../example_networks/example_network.mat")
graph = graph['network']
node_names = list(range(graph.shape[1]))
SCD_detector = SCD.SCD_obj(graph,node_names=node_names) #instantiate

## all hyperparameters
SCD_detector.list_arguments()

## set hyperparameters
param2 = {"verbose":True,"parallel_step":8}
communities = SCD_detector.detect_communities(**param2)
print(communities)
