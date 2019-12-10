### load and plot nicely.
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    p_file = "./exhaustive.pickle"

    with open(p_file, "rb") as input_file:
        out_struct = pickle.load(input_file)
    sns.set_style("whitegrid")
    krange = list(out_struct.keys())
    colors = sns.color_palette("hls", 7)
    shapes = []
    mkx = ['o']+['X']*10#('o','8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X')
    
    glopt = 0
    styles = ['-']+[":"]*10
#    fig, axs = plt.subplots()
    for enx, j in enumerate(krange):
        data = out_struct[j]
        color = colors[enx]
        marker = mkx[enx]
        x = [x[0] for x in data]
        y = [x[1] for x in data]
        y = (y - np.min(y))/(np.max(y) - np.min(y))
        max_im = np.argmax(y)
        max_point_x = x[max_im]
        max_point_y = y[max_im]
        if max_point_y > glopt:
            glopt = max_point_y
        plt.scatter(max_point_x,max_point_y,label=r"max; $\gamma$ = {}".format(j),c = color, marker = marker, s=300)
        if enx < 2:
            size = 12
        else:
            size = 80
        if enx <2:
            alpha = 0.1
        else:
            alpha = 1
        plt.plot(x,y,label=r"$\gamma$ = {}".format(j), c = color, marker = marker,linestyle=styles[enx], alpha=alpha)
    plt.legend()
    plt.tight_layout()
    plt.axhline(glopt,c="black", linestyle = "--")
    plt.text(100, glopt+0.005,"Global optimum")
#    plt.xscale('log',basex=10) 
    plt.xlabel("Number of communities (k)")
    plt.ylabel("Silhouette score")
    plt.show()
