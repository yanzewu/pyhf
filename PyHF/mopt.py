
import numpy as np

from . import rci

def rmp2_energy(n_filled_orbital, C, v, E):
    
    N = len(C)
    v_so = np.zeros((N, N, N, N))

    for i in range(N):
        for j in range(N):
            vij = np.zeros((N, N))
            for k in range(N):
                for l in range(N):
                    vij[k, l] = C[:, i].T.dot(v[:,:,k,l].dot(C[:, j]))

            v_so[i, j, :, :] = C.T.dot(vij.dot(C))

    Ecorr = 0.0

    for orbital in rci.orbital_pairs_rci(len(C), n_filled_orbital, 'd', include_ground=False):
        s = orbital.formula
        Ecorr += v_so[s[0], s[2], s[1], s[3]]*(2*v_so[s[2],s[0],s[3],s[1]] - v_so[s[2],s[1],s[3],s[0]]) \
            / (E[s[0]] + E[s[1]] - E[s[2]] - E[s[3]])

    return Ecorr
