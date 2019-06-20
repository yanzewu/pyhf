
import numpy as np
import scipy as sp

import itertools

class MolOrbital:
    """ Representation of an excited molecular orbital;
    """

    def __init__(self, formula=(), spinup=None):
        self.formula = formula
        self.spinup = spinup

    def spin_combinations(self):
        spin_orbitals = []

        for o in itertools.product('01', repeat=len(self.formula)):
            newspin = tuple((s == '0' for s in o))
            spin_orbitals.append(MolOrbital(self.formula, newspin))
        return spin_orbitals

    def level(self):
        return len(self.formula)//2


class MixedMolOrbital:
    """ Molecular orbital with mixed spin
    """
    
    def __init__(self, degen, ms, *mol_orbitals):
        self.degen = degen
        self.ms = ms
        self.spatial = mol_orbitals[0].formula
        self.spins = []
        self.coeff = np.zeros(len(mol_orbitals)//2)
        for i in range(len(mol_orbitals)//2):
            self.spins.append(mol_orbitals[2*i].spinup)
            self.coeff[i] = mol_orbitals[2*i+1]

    def level(self):
        return len(self.spatial)//2


def orbital_pairs_rci(n_orbital, n_filled_orbital, level='s', include_ground=True):
    """ Create orbital tuples;
        n_orbital and n_filled_orbital refer to spatial orbital.
        level = 's'/'d'/'sd' (single/double/single+double)
    """
    mol_orbitals = []
    if include_ground:
        mol_orbitals.append(MolOrbital())

    for l in level:
        if l == 's':
            mol_orbitals += [MolOrbital((i, a)) for i in range(n_filled_orbital) for a in range(n_filled_orbital, n_orbital)]
        elif l == 'd':
            mol_orbitals += [MolOrbital((i1, i2, a1, a2)) for i1 in range(n_filled_orbital) for i2 in range(i1+1) \
                for a1 in range(n_filled_orbital, n_orbital) for a2 in range(n_filled_orbital, a1+1)]
    return mol_orbitals


def rci_Hamiltonian(mixed_mol_orbitals, n_filled_orbital, C, h, v):
    """ Generate Hamiltonian for restricted CI;
    mol_orbitals: tuple representing excitation;
    """

    # WARNING: Only works for single excitations

    def _V2e(p, q):
        return np.sum((2*np.diag(v_so[p,q,:,:]) - np.diag(v_so[p,:,:,q]))[:n_filled_orbital])

    def _H_ss(mo1, mo2):    # H term between 2 single excitations
        
        if mo1.degen != mo2.degen:
            return 0.0

        if mo1.ms != mo2.ms:
            return 0.0

        H0 = 0.0
        s1 = mo1.spatial
        s2 = mo2.spatial

        if s1[0] == s2[0]:
            H0 += h_so[s1[1], s2[1]] + _V2e(s1[1], s2[1])           
        if s1[1] == s2[1]:
            H0 -= h_so[s2[0], s1[0]] + _V2e(s2[0], s1[0])
        if s1 == s2:
            H0 += E0    # HF energy

        # spin part
        if mo1.degen == 3:
            return H0 - v_so[s1[1], s2[1], s2[0], s1[0]]
        elif mo1.degen == 1:
            return H0 + 2*v_so[s1[1], s1[0], s2[0], s2[1]] - v_so[s1[1], s2[1], s2[0], s1[0]]

    # construct single orbital matrices

    N = C.shape[0]

    h_so = C.T.dot(h.dot(C))    # should be conjugate transpose
    v_so = np.zeros((N, N, N, N))

    for i in range(N):
        for j in range(N):
            vij = np.zeros((N, N))
            for k in range(N):
                for l in range(N):
                    vij[k, l] = C[:, i].T.dot(v[:,:,k,l].dot(C[:, j]))

            v_so[i, j, :, :] = C.T.dot(vij.dot(C))


    # Hartree Fock Energy
    E0 = 2*np.sum(np.diag(h_so)[:n_filled_orbital]) + sum(_V2e(i, i) for i in range(n_filled_orbital))

    # full Hamiltonian

    H = np.zeros((len(mixed_mol_orbitals), len(mixed_mol_orbitals)))

    for i in range(len(mixed_mol_orbitals)):
        for j in range(i+1):
  
            if mixed_mol_orbitals[i].level() == 1 and mixed_mol_orbitals[j].level() == 1:
                H[i, j] = _H_ss(mixed_mol_orbitals[i], mixed_mol_orbitals[j])
            elif mixed_mol_orbitals[i].level() == 0 and mixed_mol_orbitals[j].level() == 1:
                pass    # Brillouin Theorem
            elif mixed_mol_orbitals[i].level() == 1 and mixed_mol_orbitals[j].level() == 1:
                pass
            elif mixed_mol_orbitals[i].level() == 0 and mixed_mol_orbitals[j].level() == 0:
                H[i, j] = E0
              
    # flip
    H = H + H.T - np.diag(np.diag(H))
    return H


def adapt_spin_rci(mol_orbitals, include_degen=False):
    """ Expand the Hamiltonian, put spin in
    """
    # WARNING: Only works for single excitations.

    mixed_orbitals = []
    for m in mol_orbitals:
        if m.level() == 0:
            mixed_orbitals.append(MixedMolOrbital(1, 0, m, 1.0))
        sc = m.spin_combinations()
        if m.level() == 1:
            mixed_orbitals.append(MixedMolOrbital(1, 0, sc[0], np.sqrt(0.5), sc[3], np.sqrt(0.5)))
            mixed_orbitals.append(MixedMolOrbital(3, 0, sc[0], np.sqrt(0.5), sc[3], -np.sqrt(0.5)))

            if include_degen:
                mixed_orbitals.append(MixedMolOrbital(3, -1, sc[1], 1.0))
                mixed_orbitals.append(MixedMolOrbital(3, 1, sc[2], 1.0))

    return mixed_orbitals


def group_orbitals_by_degen(mixed_mol_orbitals):

    return [[m for m in mixed_mol_orbitals if m.degen == i+1] for i in range(4)]

def sort_orbital_by_degen(mixed_mol_orbitals):

    return sorted(mixed_mol_orbitals, key=lambda x:x.degen*10+x.ms)

def diagonalize(name, matrix, *outputs, **kwargs):
    
    eigvals = kwargs.get('n_roots', None)
    if eigvals:
        eigvals = (0, eigvals)

    E, V = sp.linalg.eigh(matrix, eigvals=eigvals)

    return (name, E, V, matrix) + outputs


def rci(n_filled_orbital, C, S, h, v, level='s', degeneracy='st', n_roots=None, external_flags=None):
    """ Performing restricted configuration interaction calculation (RCI).
    Args:
        n_filled_orbital: Number of orbitals filled;
        C: HF coefficient of basis;
        S, h, v: Integrals;
        level: 's'/'d'/'sd';
        degeneracy: 's'/'t'/'st'/'full'
    """

    mol_orbitals = orbital_pairs_rci(C.shape[0], n_filled_orbital, level)   # mol orbitals with no spin

    dtasks = []

    if degeneracy == 'full':
        mixed_mol_orbitals = adapt_spin_rci(mol_orbitals, include_degen=True)
        mixed_mol_orbitals = sort_orbital_by_degen(mixed_mol_orbitals)

        H = rci_Hamiltonian(mixed_mol_orbitals, n_filled_orbital, C, h, v)
        dtasks.append((None, H, mixed_mol_orbitals))
        
    else:

        mixed_mol_orbitals = adapt_spin_rci(mol_orbitals)
        orbital_groups = group_orbitals_by_degen(mixed_mol_orbitals)
        degen2idx = {'s':0, 'd':1, 't':2}

        for d in degeneracy:
            idx = degen2idx[d]
            H = rci_Hamiltonian(orbital_groups[idx], n_filled_orbital, C, h, v)
            dtasks.append((d, H, orbital_groups[idx]))

    return [diagonalize(*task, n_roots=n_roots) for task in dtasks]

