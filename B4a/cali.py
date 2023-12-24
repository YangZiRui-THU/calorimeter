from utils import reader_csv, get_exp_info, gaus_fit
import numpy as np
import matplotlib.pyplot as plt
import utils
import json
Energy = [1,2,5,10,50,100] # GeV
num = 1000
path = [f"./data/B4_{e}_1000.csv" for e in Energy]
edep = np.zeros(len(Energy), dtype=np.float64)
energys = np.zeros((len(Energy), num), dtype=np.float64)
center = np.zeros(len(Energy), dtype=np.float64)
sigma = np.zeros(len(Energy), dtype=np.float64)

cali_result = {}
if __name__ == '__main__':
    # raw data proess and fit
    for i,(e, p) in enumerate(zip(Energy, path)):
        egap, labs, lgap, lsen, eSen, eAbs = reader_csv(p)
        energys[i,:], pos = get_exp_info(eSen)
        mu, sig = gaus_fit(energys[i,:])
        edep[i], _ = mu*1e-3, sig*1e-3
    p = np.polyfit(edep,Energy, 1)
    cali_result['ec'] = {
        "slope": p[0],
        "intercept": p[1]}
    # corr raw data with ec linear relationship
    energys_corr = np.polyval(p, energys)
    for i,e in enumerate(Energy):
        energy_cur = energys_corr[i,:]
        mu, sig = gaus_fit(energy_cur)
        center[i], sigma[i] = mu, sig
    fig,axs = plt.subplots(2,2,figsize=(10,10))
    ax:plt.Axes = axs[0,0]
    ax.plot(edep,Energy, 'o', label="data")
    ax.plot(edep, np.polyval(p, edep), label=f"fit:{p[0]:.2f}x+{p[1]:.2f}")
    ax:plt.Axes = axs[0,1]
    ax.plot(center*1e-3,Energy, 'o', label="data")
    ax.plot(Energy,Energy)
    a,b = utils.res_fit(sigma,center,ax=axs[1,0])
    cali_result['res'] = {
        "a": a,
        "b": b}
    with open('./data/cali_result.json', 'w') as f:
        json.dump(cali_result, f)
    plt.legend()
    plt.show()