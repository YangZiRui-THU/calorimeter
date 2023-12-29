import utils
import numpy as np
import matplotlib.pyplot as plt
import json
import lmfit
from sklearn.cluster import KMeans
import pandas as pd
path = "./data/pi0.csv"
cali_result = "./data/cali_result.json"

def pi_mass(e1,p1,e2,p2):
    p0 = np.array([0,0,-500],dtype=np.float64)
    p1,p2 = np.append(p1,[0]), np.append(p2,[0])
    p1,p2 = p1-p0, p2-p0
    cos = np.dot(p1,p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))
    mass = np.sqrt(2*e1*e2*(1-cos))
    return mass
def enr_rebuild():
    global cali_result
    with open(cali_result, 'r') as f:
        cali_result = json.load(f)
    return lambda e: e*cali_result['ec']['slope'] + cali_result['ec']['intercept']

def data_filt(mass_all, dis_all):
    # return ykmeans with selected to be 1
    kmeans = KMeans(n_clusters=2,n_init='auto')
    X = np.array([dis_all, mass_all]).T
    kmeans.fit(X)
    ykmeans = kmeans.predict(X)
    # convert dtype to boolean
    ykmeans = ykmeans == 1
    if np.mean(mass_all[ykmeans]) < np.mean(mass_all[~ykmeans]):
        ykmeans = ~ykmeans
    return ykmeans

if __name__ == "__main__":
    egap, labs, lgap, lsen, eSen, eAbs = utils.reader_csv(path)
    # change unit to GeV
    # eSen = enr_rebuild()(eSen*1e-3)
    rebuild = enr_rebuild()
    num = eSen.shape[0]
    data = np.zeros(num, dtype=[('mass', 'f'), ('dis', 'f'), ('label', 'b'),('e1','f'),('e2','f'),('p1','f',(2,)),('p2','f',(2,))])
    for i in range(num):
        # shower with max energy
        idx1 = np.argmax(eSen[i])
        energy1, pos1 = utils.get_shower_info(eSen[i], idx_max=idx1)
        # shower with max energy out of shower1
        e = np.copy(eSen[i])
        e[utils.find_shower_pixel(idx1)] = 0
        idx2 = np.argmax(e)
        del e
        dis = utils.distance(utils.pixel_pos_list()[idx1], utils.pixel_pos_list()[idx2])
        energy2, pos2 = utils.get_shower_info(eSen[i], idx_max=idx2)
        energy1, energy2 = rebuild(energy1*1e-3), rebuild(energy2*1e-3)
        mass = pi_mass(energy1, pos1, energy2, pos2)
        data[i] = (mass, dis, 0, energy1, energy2, pos1, pos2)
        eSen[i][idx1] = 50
        eSen[i][idx2] = 50
    
    # data = data[data['label']]
    mass_all = data['mass']
    dis_all = data['dis']
    data['label'] = data_filt(data['mass'], data['dis'])
    data_raw = np.array([dis_all, mass_all]).T

    print(f"num: {num}")
    print(f"num mass: {np.shape(mass_all)}")
    print(f"num dis: {np.shape(dis_all)}")
    print(f"num of mass > 70:{mass_all[mass_all>70].shape}")
    print(f"num of mass < 70:{mass_all[mass_all<70].shape}")
    print(f"num of dis > 7:{dis_all[dis_all>7].shape}")
    print(f"num of dis < 7:{dis_all[dis_all<7].shape}")
    idx = dis_all>0
    mass_all = mass_all[idx]
    dis_all = dis_all[idx]
    mass_his, mass_bins = np.histogram(mass_all,bins=50)
    mass_bins = 0.5*(mass_bins[1:]+mass_bins[:-1])
    mod = lmfit.models.GaussianModel()
    par = mod.make_params(amplitude=1, center=0.13, sigma=1)
    out = mod.fit(mass_his, par, x=mass_bins)
    print(out.fit_report())
    fig,axs = plt.subplots(3,2)
    axs[0,0].hist(mass_all, bins=100)
    axs[0,0].set_title("mass distribution")
    
    out.plot_fit(axs[0,1])
    axs[0,1].set_title("fit plot")
    
    axs[1,0].hist(dis_all,bins=50)
    axs[1,0].set_title("dis distribution")
    
    axs[1,1].hist(data['e1']+data['e2'],bins=100)
    axs[1,1].set_title("e1+e2 distribution")
    
    scatter_plot = axs[2,0].scatter(mass_all, dis_all, c=data['e1']+data['e2'], s=20, cmap='viridis')
    plt.colorbar(scatter_plot, ax=axs[2,0])
    axs[2,0].set_title("mass vs dis")
    
    scatter_plot = axs[2,1].scatter(data_raw[:,1], data_raw[:,0], c=data['label'], s=20, cmap='viridis')
    plt.colorbar(scatter_plot, ax=axs[2,1])
    axs[2,0].set_title("mass vs dis")

    plt.tight_layout()
    # plt.savefig("./data/pi0.png")
    plt.show()
    fig,axs = plt.subplots(2,2)
    idx = 0
    utils.edep_plot(eSen[0], axs[0,0])
    axs[0,0].set_title(f"dis:{data_raw[idx,0]}, mass:{data_raw[idx,1]}")
    idx = 1
    utils.edep_plot(eSen[1], axs[0,1])
    axs[0,1].set_title(f"dis:{data_raw[idx,0]}, mass:{data_raw[idx,1]}")
    idx = np.argmax(data_raw[:,0])
    print(idx)
    utils.edep_plot(eSen[idx], axs[1,0])
    axs[1,0].set_title(f"dis:{data_raw[idx,0]}, mass:{data_raw[idx,1]}, e1+e2={data['e1'][idx]+data['e2'][idx]}")
    idx = np.argmin(data_raw[:,0])
    print(idx)
    utils.edep_plot(eSen[idx], axs[1,1])
    axs[1,1].set_title(f"dis:{data_raw[idx,0]}, mass:{data_raw[idx,1]}, e1+e2={data['e1'][idx]+data['e2'][idx]}")
    plt.show()