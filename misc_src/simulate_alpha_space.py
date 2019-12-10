#!/usr/bin/python

import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from scipy.optimize import curve_fit
sns.set_style("whitegrid")
points = []
for K in tqdm.tqdm(np.arange(10,5000,5)):
    for k in np.arange(10,1000,5):
        for N in [500,1000,2500,5000,10000]:
            A = 1/(12 * K * (K * N))
            B = (k + N - 1/2)/(K*(K+N))
            func = lambda tau : -1/tau**2 + 2*A*tau + B
            taus = []
            for j in [10,500,100]:
                tau_initial_guess = 1
                tau_solution = fsolve(func, tau_initial_guess)
                taus.append(tau_solution)
            tau_solution = np.mean(taus)
            points.append({"k":k,"alpha":tau_solution,"N":N,"K":K})
pfx = pd.DataFrame(points)
pfx['N'] = pfx.N.astype(str)
pfx.to_csv("simulation_data.tsv",sep="\t")
plt.savefig("simulation.pdf",dpi=300)
pfx = pd.read_csv("simulation_data.tsv",sep="\t")
sns.lineplot(pfx.K,pfx.alpha,  label = "Simulated data")

def fit_func(x, a, b):
    return a*np.power(x,2/3) + b

params = curve_fit(fit_func, pfx.K, pfx.alpha)
a, b = params[0]

X_space = pfx.K
Y_space = a * np.power(X_space,2/3) + b

plt.plot(X_space,Y_space, label = r"$"+str(np.round(a,2))+r" \cdot K^{\frac{2}{3}} "+str(np.round(b,2))+r"$")
plt.legend()
plt.ylabel(r"$\gamma$")
#plt.title(r"Fitting $\gamma$ estimate to the space of 250,000 networks")
plt.savefig("simulation_final.pdf",dpi=300)
