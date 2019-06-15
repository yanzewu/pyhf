
import numpy as np
import scipy as sp

class MolOrbital:
    """ Representation of an excited molecular orbital;
    """

    def __init__(self, formula=(), spinup=None):
        self.formula = formula
        self.spinup = spinup

    def set_spin(self, spinup):
        self.spinup = spinup

    def spin_combinations(self):
        spin_orbitals = []
        for i in 2**len(self.formula):
            newspin = tuple((s == '0' for s in bin(i)[2:]))
            spin_orbitals.append(MolOrbital(self.formula, newspin))
        return spin_orbitals

    def has_spin(self):
        return self.spinup != None

    def level(self):
        return len(self.formula)//2

class MixedMolOrbital:
    """ Molecular orbital with mixed spin
    """
    
    def __init__(self, degen, *mol_orbitals):
        self.degen = degen
        self.spatial = mol_orbitals[0].formula
        self.spins = []
        self.coeff = np.zeros(len(mol_orbitals)//2)
        for i in range(len(mol_orbitals)//2):
            self.spins.append(mol_orbitals[2*i].spinup)
            self.coeff[i] = mol_orbitals[2*i+1]

    def level(self):
        return len(self.spatial)//2


def orbital_pairs_rci(n_orbital, n_filled_orbital, level='s'):
    """ Create orbital tuples;
        n_orbital and n_filled_orbital refer to spatial orbital.
        level = 's'/'d'/'sd' (single/double/single+double)
    """
    mol_orbitals = []
    for l in level:
        if l == 's':
            mol_orbitals += [MolOrbital((i, a)) for i in range(n_filled_orbital) for a in range(n_filled_orbital, n_orbital)]
        elif l == 'd':
            mol_orbitals += [MolOrbital((i1, i2, a1, a2)) for i1 in range(n_filled_orbital) for i2 in range(i1) \
                for a1 in range(n_filled_orbital, n_orbital) for a2 in range(n_filled_orbital, a1)]
    return mol_orbitals


def create_rci_Hamiltonian(mixed_mol_orbitals, n_filled_orbital, C, h, v):
    """ Generate Hamiltonian for restricted CI;
    mol_orbitals: tuple representing excitation;
    """

    # WARNING: Only works for single excitations

    def _E_HF(orbitals):
        E = 0.0
        for i in orbitals:
            E += h_so[i, i]
        for i in orbitals:
            for j in orbitals:
                E += 2*v_so[i,i,j,j] - v_so[i,j,j,i]
        return E

    def _H_ss(mo1, mo2, h_so, v_so):

        H0 = 0.0
        s1 = mo1.spatial
        s2 = mo2.spatial

        if s1[0] == s2[0]:
            H0 += h_so[s1[1], s2[1]]
            for m in range(n_filled_orbital):
                H0 += 2*v_so[s1[1], m, s2[1], m] - v_so[s1[1], m, m, s2[1]]
                
        if s1[1] == s2[1]:
            H0 += h_so[s1[0], s2[0]]
            for m in range(n_filled_orbital):
                H0 += 2*v_so[s1[0], m, s2[0], m] - v_so[s1[0], m, m, s2[0]]
                
        # HF energy
        if s1 == s2:
            mo1seq = np.arange(n_filled_orbital)
            mo1seq[s1[0]] = s1[1]
            H0 += _E_HF(mo1seq)

        # spin part
        if mo1.degen != mo2.degen:
            return H0
        elif mo1.degen == 3:
            return H0 - v_so[s1[1], s2[1], s2[0], s1[0]]
        elif mo1.degen == 1:
            return H0 + v_so[s1[1], s1[0], s2[0], s2[1]] - v_so[s1[1], s2[1], s2[0], s1[0]]

    # construct single orbital matrices

    N = C.shape[0]

    h_so = np.zeros((N, N))
    v_so = np.zeros((N, N, N, N))

    for i in range(N):
        for j in range(N):
            h_so[i, j] = C[:, i].T.dot(h.dot(C[:, j]))

    for i in range(N):
        for j in range(N):
            vij = np.zeros((N, N))
            for k in range(N):
                for l in range(N):
                    vij[k, l] = C[:, i].T.dot(v[:,:,k,l].dot(C[:, j]))

            for k in range(N):
                for l in range(N):
                    v_so[i, j, k, l] = C[:, k].T.dot(vij.dot(C[:, l]))

    # full Hamiltonian

    H = np.zeros((len(mixed_mol_orbitals), len(mixed_mol_orbitals)))

    for i in range(len(mixed_mol_orbitals)):
        for j in range(i):
  
            if mixed_mol_orbitals[i].level() == 1 and mixed_mol_orbitals[j].level() == 1:
                H[i, j] = _H_ss(mixed_mol_orbitals[i], mixed_mol_orbitals[j], h_so, v_so)
            elif mixed_mol_orbitals[i].level() == 0 and mixed_mol_orbitals[j].level() == 1:
                pass    # Brillouin Theorem
            elif mixed_mol_orbitals[i].level() == 1 and mixed_mol_orbitals[j].level() == 1:
                pass
            elif mixed_mol_orbitals[i].level() == 0 and mixed_mol_orbitals[j].level() == 0:
                H[i, j] = _E_HF(np.arange(n_filled_orbital))
              
    # flip
    H = H + H.T - np.diag(H)
    return H


def adapt_spin_rci(mol_orbitals):
    """ Expand the Hamiltonian, put spin in
    """
    # WARNING: Only works for single excitations.

    mixed_orbitals = []
    for m in mol_orbitals:
        sc = m.spin_combinations()
        if m.level() == 1:
            mixed_orbitals.append(MixedMolOrbital(1, sc[0], np.sqrt(0.5), sc[3], np.sqrt(0.5)))
            mixed_orbitals.append(MixedMolOrbital(3, sc[0], np.sqrt(0.5), sc[3], -np.sqrt(0.5)))
            mixed_orbitals.append(MixedMolOrbital(3, sc[1], 1.0))
            mixed_orbitals.append(MixedMolOrbital(3, sc[2], 1.0))

    return mixed_orbitals


def group_orbitals_by_degen(mixed_mol_orbitals):

    return ([m for m in mixed_mol_orbitals if m.degen() == i] for i in range(4))


def rci(n_filled_orbital, C, S, h, v, level='s', degeneracy='st'):

    mol_orbitals = orbital_pairs_rci(C.shape[0], n_filled_orbital, level)   # mol orbitals with no spin
    mixed_mol_orbitals = adapt_spin_rci(mol_orbitals)

    if degeneracy == 'full':
        pass

    orbital_groups = group_orbitals_by_degen(mixed_mol_orbitals)
    degen2idx = {'s':0, 'd':1, 't':2}

    datasets = []

    for d in degeneracy:
        idx = degen2idx[d]
        H = create_rci_Hamiltonian(orbital_groups[idx], n_filled_orbital, C, h, v)
        [E, V] = np.linalg.eigh(H)
        datasets.append((d, E, V, orbital_groups[idx]))

    return datasets

