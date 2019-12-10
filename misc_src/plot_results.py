
## basic result ploting
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_result_file(fname):

    rdf = pd.DataFrame()
    with open(fname) as fn:
        for line in fn:
            line = line.strip().split("\t")
            algo_string = line[1]
            algo_name = line[4]
            ncom = line[5]
            modularity = float(line[6])
            
            nmi = float(line[7])
            ari = float(line[8])

            try:
                algo_parts = algo_string.split("__")
                number_of_nodes = algo_parts[0]
                average_deg = algo_parts[1]
                max_deg = algo_parts[2]
                expon = algo_parts[3]
                exp_com = algo_parts[4]
                mixing = algo_parts[5]            
                row = {"Algorithm":algo_name,
                       "Algo string" : algo_string,
                       "Number of communities": ncom,
                       "Modularity":modularity,
                       "NMI":nmi,
                       "ARI":ari,
                       "Number of nodes": number_of_nodes,
                       "Average degree":average_deg,
                       "Maximum degree":max_deg,
                       "Exponent":expon,
                       "Exponent community":exp_com,
                       "Mixing":mixing}
                rdf = rdf.append(row,ignore_index=True)
            except:
                row = {"Algorithm":algo_name,
                       "Number of communities": ncom,
                       "Modularity":modularity,
                       "Algo string" : algo_string,
                       "NMI":nmi,
                       "ARI":ari,
                       "Number of nodes": None,
                       "Average degree":None,
                       "Maximum degree":None,
                       "Exponent":None,
                       "Exponent community":None,
                       "Mixing":None}
                rdf = rdf.append(row,ignore_index=True)
                
        return rdf
 
    
def some_basic_plots(rframe):

    sns.boxplot(rframe.Algorithm,rframe.Modularity)
    plt.show()
    sns.violinplot(rframe.Mixing,rframe.NMI,hue=rframe.Algorithm)
    plt.show()
    sns.violinplot(rframe['Number of nodes'],rframe.NMI,hue=rframe.Algorithm)
    plt.show()
    sns.violinplot(rframe['Number of nodes'],rframe.ARI,hue=rframe.Algorithm)
    plt.show()

def get_json_data(jfile):
    with open(jfile, 'r') as f:
        return json.load(f)

def get_gt_number(fname):

    coms = []
    with open("LFR/{}/community.dat".format(fname)) as cf:
        for line in cf:
            line = line.strip().split()
            coms.append(line[1])
    return len(set(coms))

if __name__ == "__main__":

    result_file = "./results/merged2.tsv"
    result_frame = read_result_file(result_file)
    print(result_frame.columns)

    #    grouped_first = result_frame.groupby(["Algorithm","Mixing"])["ARI","NMI"].mean().reset_index()
    #    grouped_first.to_latex("./tables/latex_results.tex",index=False)
    
    result_frame['Algorithm'] = result_frame['Algorithm'].replace("EBC","SCD (defaults) - NetMF (ours)")
    result_frame['Algorithm'] = result_frame['Algorithm'].replace("EBCt","SCD - NetMF (ours)")
    result_frame['Algorithm'] = result_frame['Algorithm'].replace("NoRC","SCD - PPR (ours)")

    community_sizes_gt = []
    community_sizes_det = []
    community_algo = []
    for idx, row in result_frame.iterrows():
        gtn = get_gt_number(row['Algo string'])
        csg = row['Number of communities']
        aln = row['Algorithm']
        if aln == "Infomap":
            aln = "InfoMap"
        community_sizes_gt.append(gtn)
        community_sizes_det.append(csg)
        community_algo.append(aln)
    dfx = pd.DataFrame()
    dfx['Number of ground truth communities'] = community_sizes_gt
    dfx['Detected number'] = community_sizes_det
    dfx['Algorithm'] = community_algo
    gtns = dfx[dfx['Algorithm'] == "SCD - NetMF (ours)"]['Number of ground truth communities'].values
    dfx = dfx[dfx['Number of ground truth communities'].isin(gtns)]
    print(dfx.shape)
    dfx['Count difference'] = dfx['Detected number'].astype(int) - dfx['Number of ground truth communities'].astype(int)    
    dfx['RAc'] = dfx['Count difference'].abs()/dfx['Number of ground truth communities']
    dfx = dfx.dropna()
    dfx = dfx.groupby(["Number of ground truth communities","Count difference",'Algorithm']).mean().reset_index()
    sns.set_style("whitegrid")
    #dfx = dfx[dfx['Number of ground truth communities'] > 10]
    for alx in dfx.Algorithm.unique():
        sns.scatterplot(dfx[dfx['Algorithm'] == alx]['Number of ground truth communities'],dfx[dfx['Algorithm'] == alx]['Count difference'],label=alx)
        plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.show()
    
    grouped_first = result_frame.groupby(["Mixing","Algorithm"]).agg({'NMI':['mean','std'],'ARI':['mean','std'],"Modularity":['mean','std']}).reset_index().round(3)

    new_frame = pd.DataFrame()
    c1 = grouped_first['NMI']['mean'].astype(str)+" REPLACE "+grouped_first['NMI']['std'].astype(str)
    c2 = grouped_first['ARI']['mean'].astype(str)+" REPLACE "+grouped_first['ARI']['std'].astype(str)
    c3 = grouped_first['Modularity']['mean'].astype(str)+" REPLACE "+grouped_first['Modularity']['std'].astype(str)
    c4 = grouped_first['Algorithm']

    new_frame['Mixing'] = grouped_first['Mixing']
    new_frame['Algorithm'] = c4
    new_frame['NMI'] = c1
    new_frame['ARI'] = c2
    new_frame['Modularity'] = c3    
    print(new_frame)
    new_frame.to_latex("./tables/latex_results.tex",index=False)
    
    # some_basic_plots(result_frame)
    traces_folder = "traces/*"
    plx = sns.color_palette("cubehelix", 50)
    for enx, fix in enumerate(glob.glob(traces_folder)):
        data = get_json_data(fix)
        xs = []
        ys = []
        for k,v in data.items():
            xs.append(k)
            ys.append(v)
        plt.plot(xs,ys,color=plx[len(xs)],alpha=0.5)
    plt.xlabel("Number of clusters")
    plt.xticks(rotation=70)
    plt.ylabel("Silhouette value")
    plt.savefig("figures/landscape.png",dpi=300)
