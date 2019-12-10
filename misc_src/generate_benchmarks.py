## generate a series of benchmark graphs
import os
import subprocess


def candidate_network(a,b,c,d,e,f):
    initial_path = "./LFR/initial"
    graph_id = "__".join(str(x) for x in [a,b,c,d,e,f])
    return initial_path.replace("initial",graph_id)


def generate_network(a,b,c,d,e,f):
    params = "\n".join(str(x) for x in [a,b,c,d,e,f])
    initial_path = "LFR/initial"
    graph_id = "__".join(str(x) for x in [a,b,c,d,e,f])
    cmd = ["cp","-rvf" ,initial_path,initial_path.replace("initial",graph_id)]
    res = subprocess.check_output(cmd)
    text_file = open(initial_path.replace("initial",graph_id)+"/"+"parameters.dat", "w")
    text_file.write(params)
    text_file.close()
    cmd2 = "cd "+initial_path.replace("initial",graph_id)+";./benchmark"
    print(cmd2)
    os.system(cmd2)
    return initial_path.replace("initial",graph_id)

if __name__ == "__main__":
    import argparse
    #     1000	# number of nodes
    # 15	# average degree
    # 50	# maximum degree
    # 2	# exponent for the degree distribution
    # 1	# exponent for the community size distribution
    # 0.2	# mixing parameter
    # 20	# minimum for the community sizes (optional; just comment this line, if you wish)
    # 50	# maximum for the community sizes (optional; just comment this line, if you wish)

    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_nodes",default=1000)
    parser.add_argument("--average_degree",default=10)
    parser.add_argument("--maximum_degree",default=10)
    parser.add_argument("--expon",default=2)
    parser.add_argument("--exp_com",default=1)
    parser.add_argument("--mparam",default=0.2)
    parser.add_argument("--task",default="EBC")
    parser.add_argument("--networks",default="simulated")
    args = parser.parse_args()

    a = args.number_of_nodes
    b = args.average_degree
    c = args.maximum_degree
    d = args.expon
    e = args.exp_com
    f = args.mparam

    if args.networks == "simulated":
        numbers_of_nodes = [100,500,750,1000,2500,5000,10000]
        average_degree = [15,30,50]
        maximum_degree = [10,50,100]
        expon = [2,2.5,3]
        exp_com = [1]
        mparam = [0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1]

        for a in numbers_of_nodes:
            for b in average_degree:
                for c in maximum_degree:
                    for d in expon:
                        for e in exp_com:
                            for f in mparam:
                                network_folder = candidate_network(a,b,c,d,e,f)
                                try:
                                    #                                if not os.path.isfile(network_folder):
                                    #                                    continue
                                    #network_folder = generate_network(a,b,c,d,e,f)
                                    if args.task == "benchmark":
                                        benchmark_command = "python3 community_algorithms.py --network_tag SYNTHETIC --input_network_folder "+ network_folder
                                        os.system(benchmark_command)
                                    elif args.task == "EBC":
                                        benchmark_command = "python3 benchmark_ebc.py --network_tag SYNTHETIC --input_network_folder "+ network_folder
                                        os.system(benchmark_command)
                                    else:
                                        benchmark_command = "python3 community_range.py --network_tag SYNTHETIC --input_network_folder "+ network_folder
                                        os.system(benchmark_command)
                                except:
                                    pass
    else:
        for root, dirs, files in os.walk("../data"):
            for dir in dirs:
                benchmark_command = "python3 community_algorithms.py --network_tag REAL --input_network_folder ../data/"+ dir
                print(benchmark_command)
                os.system(benchmark_command)
