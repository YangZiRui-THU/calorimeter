import numpy as np
import time
import subprocess
import functools
import matplotlib.pyplot as plt
import lmfit
path = "./build/B4_1_1000.csv"
nofCells = 40
calorSizeXY = 1.01 # cm
def extract(path):
    energy = int(path.split('_')[1])
    num = int(path.split('_')[2].split('.')[0])
    return energy, num

def reader_root(path):
    import uproot
    with uproot.open(path) as f:
        tree = f['B4']
        tree.show()
        start_time = time.time()
        keys = tree.keys()
        print("keys:",time.time() - start_time)
        egap = tree.arrays(keys[0])
        labs = tree.arrays(keys[1])
        lgap = tree.arrays(keys[2])
        lsen = tree.arrays(keys[3])
        print("arrays:",time.time() - start_time)
        eSen = tree.arrays([keys[i+4] for i in range(nofCells*nofCells)])
        eAbs = tree.arrays([keys[i+nofCells*nofCells+4] for i in range(nofCells*nofCells)])
        print("arrays",time.time() - start_time)
        print(keys[4])
        print(keys[nofCells*nofCells+4])
        return egap, labs, lgap, lsen, eSen, eAbs
def run_cmd(cmd, f):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output, error = p.communicate()
    return f(output.strip())
def reader_csv(path):
    note_num = run_cmd(f"cat {path} | grep '#' | wc -l", int)
    data = np.loadtxt(path, dtype=np.float64, delimiter=',', skiprows=note_num)
    egap = data[:,0]
    labs = data[:,1]
    lgap = data[:,2]
    lsen = data[:,3]
    eSen = data[:, 4:4+nofCells*nofCells]
    eAbs = data[:, 4+nofCells*nofCells:]
    return egap, labs, lgap, lsen, eSen, eAbs

def pixel_pos(i,j):
    # return pixel center position with unit cm
    x = (i - nofCells/2. + 0.5) * calorSizeXY
    y = (j - nofCells/2. + 0.5) * calorSizeXY
    return np.array([x, y])
@functools.lru_cache(maxsize=2)
def pixel_pos_list():
    return np.array([pixel_pos(i,j) for i in range(nofCells) for j in range(nofCells)])

def distance(pos1, pos2):
    return np.linalg.norm(pos1-pos2)

@functools.lru_cache(maxsize=None)
def find_shower_pixel(idx_max,d = 3.5):
    # find pixels with distance less than 3.5cm with idx_max of pixel
    max_pos = pixel_pos_list()[idx_max]
    shower_pixels = np.array([i for i in range(nofCells*nofCells) if distance(pixel_pos_list()[i], max_pos) < d])
    return shower_pixels
def get_shower_info(eSen, idx_max=None):
    # return shower_energy, shower_pos
    if idx_max is None:
        idx_max = np.argmax(eSen)
    shower_pixels = find_shower_pixel(idx_max)
    shower_energy = np.sum(eSen[shower_pixels])
    shower_pos = np.average(pixel_pos_list()[shower_pixels], axis=0, weights=eSen[shower_pixels])
    return shower_energy, shower_pos
def get_exp_info(eSen):
    # input shape 1000*1600, all data from a certain energy
    num = eSen.shape[0]
    energys = np.zeros(num)
    pos = np.zeros((num,2))
    for i in range(num):
        energys[i],pos[i] = get_shower_info(eSen[i])
    return energys, pos
def gaus_fit(energys, ax:plt.Axes=None):
    # return center, sigma
    hist, bins = np.histogram(energys,bins=50)
    bins = 0.5*(bins[1:]+bins[:-1])
    mod = lmfit.models.GaussianModel()
    pars = mod.guess(hist, x=bins)
    out = mod.fit(hist, pars, x=bins)
    if not out.success:
        raise Exception("gaus fit failed")
    if ax is not None:
        out.plot_fit(ax)
    return out.best_values['center'], out.best_values['sigma']
def res_fit(sigma, energy, ax:plt.Axes=None):
    mod = lmfit.models.ExpressionModel("sqrt((a/sqrt(e))**2+b**2)",independent_vars=['e'])
    par = mod.make_params(a=1,b=1)
    res = mod.fit(sigma/energy, par, e=energy)
    if not res.success:
        raise Exception("res fit failed")
    if ax is not None:
        res.plot_fit(ax,numpoints=1000)
    return res.best_values['a'], res.best_values['b']
def edep_plot(eSen, ax:plt.Axes, v = None, norm=None):
    # input shape 1600,
    eSend2d = np.reshape(eSen, (nofCells,nofCells))
    if v is not None:
        h = ax.imshow(eSend2d, cmap='plasma', interpolation='nearest',norm=norm, vmin=v[0], vmax=v[1])
    else:
        h = ax.imshow(eSend2d, cmap='plasma', interpolation='nearest',norm=norm)
    return h
if __name__ == "__main__":
    energy, num = extract(path)
    egap, labs, lgap, lsen, eSen, eAbs = reader_csv(path)
    energys, pos = get_exp_info(eSen)
    fig, axs= plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"energy:{energy}MeV, num:{num}")
    ax:plt.Axes = axs[0,0]
    ax.set_title("energy histogram")
    mu, resolution = gaus_fit(energys, ax)
    ax:plt.Axes = axs[0,1]
    ax.set_title("position scatter")
    h2 = ax.scatter(pos[:,0], pos[:,1])
    ax:plt.Axes = axs[1,0]
    ax.set_title("energy deposit of last one")
    eSen2d = np.reshape(eSen, (nofCells,nofCells,num))
    h3 = ax.imshow(np.reshape(eSen[-1,:], (nofCells,nofCells)), cmap='plasma', interpolation='nearest')
    ax:plt.Axes = axs[1,1]
    ax.set_title("shower range of last one")
    shower_idx = find_shower_pixel(np.argmax(eSen[-1,:]))
    shower_idx_not = np.fromiter((i for i in range(nofCells*nofCells) if i not in shower_idx), int)
    eSen[-1,:][shower_idx_not] = 0
    h4 = ax.imshow(np.reshape(eSen[-1,:],(nofCells,nofCells)), cmap='plasma', interpolation='nearest')
    plt.show()