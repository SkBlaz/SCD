## The SCD_obj algorithm Skrlj 2019
import networkx as nx
import numpy as np
import tqdm
import multiprocessing as mp

from sklearn.metrics import silhouette_score
import scipy.sparse as sp
import scipy as spy
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import svds
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist

import sklearn.metrics.pairwise
from sklearn.cluster import MiniBatchKMeans

import torch
from collections import defaultdict
from itertools import product
import operator

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

class SCD_obj:
    def __init__(self,input_graph,verbose=True, node_names=None,input_type="sparse_matrix",device = "cpu"):

        """
        Initiator class
        """
        if input_type == "networkx":
            node_names = list(input_graph.nodes())
            input_graph = nx.to_scipy_sparse_matrix(input_graph)
            
        self.verbose = verbose
        self.device = device
        self.all_scores = None
        self.node_names = node_names
        self.cluster_quality = {}
        self.default_parameters = {"clustering_scheme":"hierarchical","max_com_num":"100","verbose":False,"sparisfy":True,"parallel_step":6,"prob_threshold":0.0005, "community_range":[1,3,5,7,11,20,40,50,100,200,300],"fine_range":3,"lag_threshold":10,"num_important":10000}
        self.input_graph = input_graph
        
    def page_rank_kernel(self,index_row):
        """
        a kernel for parallel PPRS execution
        param: index of a node.
        output: PPR vector
        """
        
        pr = self.PPRS([index_row],
                       epsilon=1e-6,
                       max_steps=100000,
                       damping=0.90,
                       spread_step=10,
                       spread_percent=0.1,
                       try_shrink=True)
        norm = np.linalg.norm(pr, 2)
        if norm > 0:
            pr = pr / np.linalg.norm(pr, 2)
            return (index_row,pr)
        else:
            return (index_row,np.zeros(self.normalized.shape[1]))

    def sparse_pr(self,max_iter=10000, tol=1.0e-6,alpha=0.85):

        """
        Sparse personalized pagerank.
        """
        
        N = self.input_graph.shape[1]
        nodelist = np.arange(self.input_graph.shape[1])
        S = spy.array(self.input_graph.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = sp.spdiags(S.T, 0, *self.input_graph.shape, format='csr')
        M = Q * self.input_graph
        x = spy.repeat(1.0 / N, N)
        p = spy.repeat(1.0 / N, N)
        dangling_weights = p
        is_dangling = spy.where(S == 0)[0]
        for _ in range(max_iter):
            xlast = x
            x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
                (1 - alpha) * p
            err = spy.absolute(x - xlast).sum()
            if err < N * tol:
                return dict(zip(nodelist, map(float, x)))

    def stochastic_normalization(self,matrix=None,return_matrix=False):

        """
        Perform stochastic normalization
        """
        
        if matrix is None:
            matrix = self.input_graph.tolil()        
        try:
            matrix.setdiag(0)
        except TypeError as te:
            matrix.setdiag(np.zeros(matrix.shape[0]))            
        matrix = matrix.tocsr()
        d = matrix.sum(axis=1).getA1()
        nzs = np.where(d > 0)
        k = 1/d[nzs]
        matrix = (sp.diags(k, 0).tocsc().dot(matrix)).transpose()
        if return_matrix:
            return matrix
        else:
            self.normalized = matrix
        
    def modularity(self, communities, weight='weight'):

        """
        Classic computation of modularity.
        """
        
        G = nx.from_scipy_sparse_matrix(self.input_graph)
        multigraph = G.is_multigraph()
        directed = G.is_directed()
        m = G.size(weight=weight)
        if directed:
            out_degree = dict(G.out_degree(weight=weight))
            in_degree = dict(G.in_degree(weight=weight))
            norm = 1 / m
        else:
            out_degree = dict(G.degree(weight=weight))
            in_degree = out_degree
            norm = 1 / (2 * m)

        def val(u, v):
            try:
                if multigraph:
                    w = sum(d.get(weight, 1) for k, d in G[u][v].items())
                else:
                    w = G[u][v].get(weight, 1)
            except KeyError:
                w = 0
            # Double count self-loops if the graph is undirected.
            if u == v and not directed:
                w *= 2
            return w - in_degree[u] * out_degree[v] * norm

        Q = np.sum(val(u, v) for c in communities for u, v in product(c, repeat=2))
        return Q * norm

    def PPRS(self, start_nodes,
                         epsilon=1e-6,
                         max_steps=100000,
                         damping=0.5,
                         spread_step=10,
                         spread_percent=0.3,
                         try_shrink = True):

        """
        Personalized PageRank with shrinking
        """
        
        assert(len(start_nodes)) > 0
        size = self.normalized.shape[0]
        if start_nodes is None:
            start_nodes = range(size)
            nz = size
        else:
            nz = len(start_nodes)
        start_vec = np.zeros((size, 1))
        start_vec[start_nodes] = 1
        start_rank = start_vec / len(start_nodes)
        rank_vec = start_vec / len(start_nodes)
        shrink = False
        which = np.zeros(0)
        if len(self.global_top_nodes) < self.normalized.shape[1] and self.sample_whole_graph == False:
            which= np.full(self.normalized.shape[1],False)
            which[self.global_top_nodes] = True
            start_rank = start_rank[which]
            rank_vec = rank_vec[which]
            self.normalized = self.normalized[:, which][which, :]
            start_vec = start_vec[which]
            size = len(self.global_top_nodes)
        else:
            which = np.zeros(0)            
        if try_shrink:
            v = start_vec / len(start_nodes)
            steps = 0
            while nz < size * spread_percent and steps < spread_step:
                steps += 1
                v += self.normalized.dot(v)
                nz_new = np.count_nonzero(v)
                if nz_new == nz:
                    shrink = True
                    break
                nz = nz_new
            rr = np.arange(self.normalized.shape[0])
            which = (v[rr] > 0).reshape(size)
            if shrink:
                start_rank = start_rank[which]
                rank_vec = rank_vec[which]
                self.normalized = self.normalized[:, which][which, :]        
        diff = np.Inf
        steps = 0
        while diff > epsilon and steps < max_steps:
            steps += 1
            new_rank = self.normalized.dot(rank_vec)
            rank_sum = np.sum(new_rank)
            if rank_sum < 0.999999999:
                new_rank += start_rank * (1 - rank_sum)
            new_rank = damping * new_rank + (1 - damping) * start_rank
            new_diff = np.linalg.norm(rank_vec - new_rank, 1)
            diff = new_diff
            rank_vec = new_rank
        if try_shrink and shrink:
            ret = np.zeros(size)        
            rank_vec = rank_vec.T[0]
            ret[which] = rank_vec
            if start_nodes[0] < len(ret):
                ret[start_nodes] = 0
            return ret.flatten()
        else:
            if start_nodes[0] < len(rank_vec):
                rank_vec[start_nodes] = 0
            return rank_vec.flatten()

    def emit(self,message):
        """
        Simple logging wrapper
        """
        logging.info(message)

    def get_sparse_walk_matrix(self,num_important,prob_threshold=0,parallel_step=6):

        """
        Get walk matrix
        """
        
        if self.verbose:
            self.emit("Walking..")

        if self.node_names is None:
            self.node_names = list(range(self.input_graph.shape[1]))
        self.global_pagerank = sorted(self.sparse_pr().items(), key=operator.itemgetter(1),reverse=True)
        self.global_top_nodes = [x[0] for x in self.global_pagerank[0:num_important]]
        self.stochastic_normalization()
        n = self.normalized.shape[1]
        edgelist_triplets = []
        jobs = [range(n)[i:i + parallel_step] for i in self.global_top_nodes]
        with mp.Pool(processes=parallel_step) as p:
            for batch in tqdm.tqdm(jobs):
                results = p.map(self.page_rank_kernel,batch)
                for nid, result_vector in results:
                    cols = np.argwhere(result_vector > prob_threshold).flatten().astype(int)
                    vals = result_vector[cols].flatten()
                    ixx = np.repeat(nid,cols.shape[0]).flatten().astype(int)
                    arx = np.vstack((ixx,cols,vals)).T                
                    edgelist_triplets.append(arx)
        sparse_edgelist = np.concatenate(edgelist_triplets,axis=0)
        print("Compressed to {}% of the initial size".format((sparse_edgelist.shape[0]*100)/n**2))
        vectors = sp.coo_matrix((sparse_edgelist[:,2], (sparse_edgelist[:,0].astype(int),sparse_edgelist[:,1].astype(int)))).tocsr()
        return vectors

    def approximate_normalized_graph_laplacian(self,A, rank, which="LA"):
        n = A.shape[0]
        L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
        X = sp.identity(n) - L
        self.emit("Eigen decomposition...")
        evals, evecs = eigsh(X, rank, which=which)
        self.emit("Maximum eigenvalue {}, minimum eigenvalue {}".format(np.max(evals), np.min(evals)))
        self.emit("Computing D^{-1/2}U..")
        D_rt_inv = sp.diags(d_rt ** -1)
        D_rt_invU = D_rt_inv.dot(evecs)
        return evals, D_rt_invU

    def approximate_deepwalk_matrix(self,evals, D_rt_invU, window, vol, b):
        evals = self.deepwalk_filter(evals, window=window)
        X = sp.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
        device = torch.device(self.device) #('cuda' if torch.cuda.is_available() else 'cpu')
        m = torch.from_numpy(X).to(device)
        vx = torch.tensor(vol/b).to(device)
        mmt = torch.mm(m,m.t()).double()
        vol = vx.expand_as(mmt).double()
        mmt2 = mmt * vol
        Y = torch.log(torch.clamp(mmt2,min = 1)).cpu()
        self.emit("Computed DeepWalk matrix with {} non-zero elements".format(np.count_nonzero(Y)))
        return sp.csr_matrix(Y)

    def svd_deepwalk_matrix(self,X, dim):
        self.emit("Computing SVD..")
        u, s, v = svds(X, dim, return_singular_vectors="u")
        return sp.diags(np.sqrt(s)).dot(u.T).T

    def deepwalk_filter(self,evals, window):
        for i in range(len(evals)):
            x = evals[i]
            evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
        evals = np.maximum(evals, 0)
        self.emit("After filtering, max eigenvalue={}, min eigenvalue={}".format(np.max(evals), np.min(evals)))
        return evals

    def netMF_large (self,A,rank=256,embedding_dimension=128,window=10,negative=1.0):

        self.emit("Running TorchNetMF for a large window size...")
        self.emit("Window size is set to be {}".format(window))

        if rank >= A.shape[1]:
            rank = A.shape[1]-1

        if embedding_dimension >= A.shape[1]:
            embedding_dimension = 8

        # load adjacency matrix
        vol = float(A.sum())
        
        # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
        # keep top #rank eigenpairs
        
        evals, D_rt_invU = self.approximate_normalized_graph_laplacian(A, rank=rank, which="LA")

        # approximate deepwalk matrix
        deepwalk_matrix = self.approximate_deepwalk_matrix(evals, D_rt_invU,
                window=window,vol=vol, b=negative)

        # factorize deepwalk matrix with SVD
        return self.svd_deepwalk_matrix(deepwalk_matrix, dim=embedding_dimension)
    
    def cross_entropy (self,data,clusters):

        """
        The symmetric cross entropy is computed as follows:
        SymCE = - sum_{i} (P(i) + Q(i)(log(P(i))+log(Q(i))

        Idea:
        for each potential cluster:
        compute intra cluster cross entropy
        Min intra, max inter.
        """

        per_cluster = defaultdict(list)        
        unique_clusters = set(clusters)
        clusters = np.array(clusters)
        internal_entropies = []
        for label in unique_clusters:
            indices= np.where(clusters==label)
            cmat = data[indices]
            flat_vals = np.array(cmat.todense().flatten())
            entropy_internal = -np.sum(np.multiply(flat_vals,np.log(flat_vals)))
            if np.isnan(entropy_internal):
                entropy_internal = 1
            internal_entropies.append(entropy_internal)
        joint_entropy = 1/np.mean(internal_entropies)
        return joint_entropy
            
    def compute_cluster_quality(self,clusters,weight="weight",optional_data=None):
                
        if self.clustering_measure == "euclidean":
            distances = cdist(clusters, clusters, 'euclidean')            
            max_dist = np.max(distances)#/np.mean(distances)
            return max_dist
        
        elif self.clustering_measure == "modularity":
             return self.modularity(clusters,weight)

        elif self.clustering_measure == "silhouette":
            return silhouette_score(optional_data, clusters)

        elif self.clustering_measure == "entropy":
            return self.cross_entropy(optional_data,clusters)
        
        else:
            self.emit("Please, select a quality measure!")
    
    def kmeans_clustering(self,vectors,community_range,stopping,improvement_step=0.001,return_score=False):
        
        if self.verbose:
            self.emit("Doing kmeans search")

        if len(community_range) == 1:
            fine_interval = 0
            
        else:
            fine_interval = int((community_range[2]-1)/2)
            community_range = np.arange(community_range[0],community_range[1],community_range[2])
        
        nopt = 0            
        lag_num = 0
        mx_opt = 0
        opt_partitions = None
        fine_range = None
        all_scores = []

        for nclust in community_range:
            dx_hc = defaultdict(list)
            clustering_algorithm = MiniBatchKMeans(n_clusters=nclust, init_size = 3*nclust)
            clusters = clustering_algorithm.fit_predict(vectors).tolist()
            for a, b in zip(clusters, self.node_names):
                dx_hc[a].append(b)
            partitions = dx_hc
            lx = np.max([len(y) for x,y in partitions.items()])
            if self.clustering_measure == "silhouette"  or self.clustering_measure == "entropy":                    
                mx = self.compute_cluster_quality(clusters,optional_data=vectors)
            else:
                mx = self.compute_cluster_quality(clustering_algorithm.cluster_centers_)
            self.cluster_quality[str(nclust)] = mx
            all_scores.append((nclust,mx))
            if mx > mx_opt+improvement_step:
                lag_num = 0
                opt_partitions = partitions
                nopt = nclust
                if self.verbose:
                    self.emit("Improved quality: {}, found {} communities. Maximum size {}".format(np.round(mx,3),nopt, lx))
                mx_opt = mx
                self.opt_clust = partitions
                
            else:
                lag_num+=1
                if self.verbose:
                    self.emit("No improvement for {} iterations".format(lag_num))
            if lag_num == stopping:
                if self.verbose:
                    self.emit("Stopping criterion reached. Fine tunning.")
                    break
                lag_num+=1

        if self.verbose:
            self.emit("Initial screening returned optimum of: {}".format(nopt))

        ## Do some fine tunning of the k
        if fine_interval > 0:
            fine_range = [x for x in np.arange(nopt-fine_interval,nopt+fine_interval,1) if x > 0]
            for cand in tqdm.tqdm(fine_range):
                dx_hc = defaultdict(list)
                clustering_algorithm = MiniBatchKMeans(n_clusters=nclust, init_size = 3*nclust)
                clusters = clustering_algorithm.fit_predict(vectors).tolist()
                for a, b in zip(clusters,self.node_names):
                    dx_hc[a].append(b)
                partitions = dx_hc
                if self.clustering_measure == "silhouette" or self.clustering_measure == "entropy":
                    mx = self.compute_cluster_quality(clusters,optional_data=vectors)
                else:
                    mx = self.compute_cluster_quality(clustering_algorithm.cluster_centers_)
                if mx > mx_opt+improvement_step:
                    lag_num = 0
                    opt_partitions = partitions
                    if self.verbose:
                        self.emit("Improved quality: {}, found {} communities.".format(np.round(mx,2),len(partitions)))
                    mx_opt = mx
                    self.opt_clust = dx_hc
                    nopt = nclust
                    
        if return_score:
            return opt_partitions,mx_opt,all_scores
        
        else:
            return opt_partitions

    def get_exp_features(self,cutoff=0.01, order = 2):

        core_graph = self.input_graph.tocsc()
        feature_matrices = [core_graph]
        for j in range(order):
            if self.verbose:
                self.emit("Order of computation: {}".format(j))
            new = sp.linalg.expm(feature_matrices[j])
            feature_matrices.append(new)
        final_matrix = feature_matrices[0]
        for en, j in enumerate(feature_matrices):
            if en > 0:
                tar = self.stochastic_normalization(j.tocsr(),return_matrix=True)
                final_matrix = final_matrix.tocsr().multiply(tar)
        return final_matrix.tocsr()

    def list_arguments(self):
        argument_list = {    
            "verbose":False,
            "sparisfy":True,
            "parallel_step":6,
            "prob_threshold (only for ppr_embedding)":0.0005,
            "community_range" : "auto",
            "num_important (only for PPR)":100,
            "clustering_measure":"silhouette",
            "stopping":5,
            "improvement_step":0.05,
            "node_feature_type (netmf_embedding or ppr_embedding)" : "netmf_embedding",
            "negative_range (negative sampling range)" : [1],
            "window_sizes (range of netmf window sizes)" : [3],
            "dims (latent dimension for netmf)" : [64],
            "output_format (grouped or nongrouped)" : "nongrouped",
            "use_normalized_scores (bool, normalize per netmf hyperparams)":True
            }
        for k,v in argument_list.items():
            print(k,"default setting:",v)
        
    
    def detect_communities(self,verbose=False,sparisfy=True,parallel_step=6,prob_threshold=0.0005, community_range = "auto", num_important=100,clustering_measure="silhouette",stopping=5,improvement_step=0.05,node_feature_type = "netmf_embedding", negative_range = [1], window_sizes = [3], dims = [64],output_format = "nongrouped", use_normalized_scores=True, custom_embedding_vectors = None):

        if community_range == "auto":
            K = self.input_graph.shape[1]
            kpow = int(0.42*np.power(K, 2/3) - 5.7)
            community_range = [kpow,K,kpow]
        
        if self.verbose:
            self.emit("Important nodes: {}".format(num_important))
        self.clustering_measure = clustering_measure
        
        ## step 1: embedding
        self.sample_whole_graph = False
        if node_feature_type == "EXP":
            vectors = self.get_exp_features()            
        elif node_feature_type == "netmf_embedding":

            ## optimize the joint space..
            lopt = 0
            best_partition = None
            self.opt_score = -1
            self.opt_k = -1
            self.opt_mean = -1
            all_scores = None
            self.opt_trace = []
            for n in negative_range:
                for w in window_sizes:
                    for d in dims:
                        if self.verbose:
                            self.emit("testing setting {} {} {}".format(n,w,d))
                        vectors = self.netMF_large(self.input_graph,embedding_dimension=d,window=w,negative=n)
                        tmp_partition, score, score_dump = self.kmeans_clustering(vectors,community_range,stopping,improvement_step,return_score=True)
                        
                        if use_normalized_scores and len(score_dump) > 1:
                            normalized_score = (score - np.min(score_dump))/(np.max(score_dump) - np.min(score_dump))
                        else:
                            normalized_score = score

                        norm_score_dump = (score_dump - np.min(score_dump))/(np.max(score_dump) - np.min(score_dump))

                        score_mean = np.mean(norm_score_dump) ## this was added after publication as it works better!
                        self.opt_trace.append({"score":normalized_score,"negative":n,"window":w,"dimension":d, "smean":score_mean})
                        if normalized_score > self.opt_score and tmp_partition and score_mean > self.opt_mean:
                            self.opt_mean = score_mean
                            self.all_scores = score_dump
                            self.opt_k = len(tmp_partition)
                            best_partition = tmp_partition
                            self.opt_score = normalized_score
                            self.opt_config = {"score":normalized_score,"negative":n,"window":w,"dimension":d,"Num communities":len(tmp_partition), "smean":score_mean}
                        else:
                            self.emit("Invalid embedding")

        elif node_feature_type == "ppr_embedding":
            vectors = self.get_sparse_walk_matrix(num_important,prob_threshold,parallel_step)
            self.emit("Starting cluster detection..")
            best_partition, score, score_dump = self.kmeans_clustering(vectors,community_range,stopping,improvement_step, return_score=True)
            self.opt_score = score

        elif node_feature_type == "custom_embedding":
            self.emit("Starting cluster detection..")
            best_partition, score, score_dump = self.kmeans_clustering(custom_embedding_vectors,community_range,stopping,improvement_step, return_score=True)
            self.opt_score = score
            
        if self.verbose:
            self.emit("Obtained vectors of shape {}".format(vectors.shape))
            self.emit(self.opt_config)

        assert vectors.shape[0] == self.input_graph.shape[0]
        
        if output_format == "grouped":
            return best_partition
        
        else:
            out_struct = {}
            for k,els in best_partition.items():
                for el in els:
                    out_struct[el] = k
            return out_struct
                        
if __name__ == "__main__":

    import scipy.io
    graph = scipy.io.loadmat("../example_networks/example_network.mat")
    graph = graph['network']
    node_names = list(range(graph.shape[1]))
    NRC= SCD_obj(graph,node_names=node_names)
    param2 = {"verbose":True,"parallel_step":16}
    communities = NRC.detect_communities(**param2)
ls 
