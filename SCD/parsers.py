import networkx as nx

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
