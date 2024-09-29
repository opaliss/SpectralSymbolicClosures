import numpy as np


def epsilon_electric(E_ROM, E_FOM, Nx, avg_time=True):
    num = np.sum(np.abs(np.abs(E_ROM[Nx:]) - np.abs(E_FOM[Nx:]))**2, axis=0)
    denom = np.sum(np.abs(E_FOM[Nx:])**2, axis=0)
    if avg_time:
        return np.mean(num/denom)
    else:
        return num/denom