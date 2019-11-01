
import numpy as np

from . import rci
from . import integration


def rci_soc_Hamiltonian(mixed_mol_orbitals, C, L):
    
    # WARNING: Only works for single excitation

    def _V(i, j, spin1, spin2):
        if spin1 and spin2:
            return Lz_so[i, j]
        elif not spin1 and not spin2:
            return -Lz_so[i, j]
        elif spin1 and not spin2:
            return Lx_so[i, j] - 1j*Ly_so[i, j]
        else:
            return Lx_so[i, j] + 1j*Ly_so[i, j]

    def _H_sg(mo1):

        # <s|H|g>

        if mo1.degen == 1:
            return 0.0
        
        H = 0.0j
        for c1, s1 in zip(mo1.coeff, mo1.spins):
            H += _V(mo1.spatial[1], mo1.spatial[0], s1[1], s1[0])*c1

        return H

    def _H_ss(mo1, mo2):

        # simple cases
        if mo1.degen == 1 and mo2.degen == 1:
            return 0.0
        elif mo1.degen == 3 and mo2.degen == 3 and mo1.ms * mo2.ms == -1:
            return 0.0

        H = 0.0j

        for c1, s1 in zip(mo1.coeff, mo1.spins):
            for c2, s2 in zip(mo2.coeff, mo2.spins):
                if mo1.spatial[0] == mo2.spatial[0] and s1[0] == s2[0]:
                    H += _V(mo1.spatial[1], mo2.spatial[1], s1[1], s2[1])*c1*c2
                if mo1.spatial[1] == mo2.spatial[1] and s1[1] == s2[1]:
                    H -= _V(mo2.spatial[0], mo1.spatial[0], s2[0], s1[0])*c1*c2

        return H
        
    # construct single orbital matrices

    Lx, Ly, Lz = L
    Lx_so = C.T.dot(Lx.dot(C))
    Ly_so = C.T.dot(Ly.dot(C))
    Lz_so = C.T.dot(Lz.dot(C))

    H = np.zeros((len(mixed_mol_orbitals), len(mixed_mol_orbitals)), dtype=complex)

    for i in range(len(mixed_mol_orbitals)):
        for j in range(i+1):
            
            if mixed_mol_orbitals[i].level() == 1 and mixed_mol_orbitals[j].level() == 1:
                H[i, j] = _H_ss(mixed_mol_orbitals[i], mixed_mol_orbitals[j])
            elif mixed_mol_orbitals[i].level() == 1 and mixed_mol_orbitals[j].level() == 0:
                H[i, j] = _H_sg(mixed_mol_orbitals[i])
            elif mixed_mol_orbitals[i].level() == 0 and mixed_mol_orbitals[j].level() == 1:
                H[i, j] = _H_sg(mixed_mol_orbitals[j]).conjugate()
            elif mixed_mol_orbitals[i].level() == 0 and mixed_mol_orbitals[j].level() == 0:
                pass

    H = H + H.T.conj() - np.diag(np.diag(H))
    return H


def rcis_with_soc(n_filled_orbital, C, S, h, v, bases, atom_coords, atom_charges, n_roots=None):
    
    # Create L matrices first
    Lx = np.zeros((len(bases), len(bases)), dtype=complex)
    Ly = np.zeros((len(bases), len(bases)), dtype=complex)
    Lz = np.zeros((len(bases), len(bases)), dtype=complex)

    for i in range(len(bases)):
        for j in range(i+1):
            lx, ly, lz = orbital_angular_momentum(bases[i], bases[j], atom_coords, atom_charges)
            Lx[i, j] = -1j*lx
            Ly[i, j] = -1j*ly
            Lz[i, j] = -1j*lz

    L = [L1+L1.T.conj()-np.diag(np.diag(L1)) for L1 in (Lx, Ly, Lz)]

    mol_orbitals = rci.orbital_pairs_rci(C.shape[0], n_filled_orbital, 's')
    mixed_mol_orbitals = rci.adapt_spin_rci(mol_orbitals, include_degen=True)
    mixed_mol_orbitals = rci.sort_orbital_by_degen(mixed_mol_orbitals)

    Hsoc = rci_soc_Hamiltonian(mixed_mol_orbitals, C, L)
    Hel = rci.rci_Hamiltonian(mixed_mol_orbitals, n_filled_orbital, C, h, v)

    Hfull = Hel - Hsoc*0.25/137.036**2
    return [rci.diagonalize(None, Hfull, mixed_mol_orbitals, n_roots=n_roots)]
    
    # _, E, V, _ = rci.diagonalize(None, Hel)
    # Hsoc_later = V.T.dot(Hsoc.dot(V))

    # cnt1, cnt3 = 0, 0.0
    # for i in range(1, len(V)):
    #     idx = np.argmax(V[:,i])
    #     d = mixed_mol_orbitals[idx].degen
    #     print('%s%d%s' % ('S' if d == 1 else 'T', cnt1+1 if d == 1 else int(cnt3)+1, ' (ms=%d)'%mixed_mol_orbitals[idx].ms if d == 3 else ''))
    #     if d == 1: cnt1 += 1
    #     else: cnt3 += 1/3

    # print('\n'.join(('\t'.join(('%.2f%s%.2fi'%(Hsoc_later[i,j].real, '+' if Hsoc_later[i,j].imag >=0.0 else '', Hsoc_later[i,j].imag) for j in range(len(Hsoc))))) for i in range(len(Hsoc))))


def orbital_angular_momentum(lhs, rhs, atom_coords, atom_charges):
    """ Calculate **REAL** OAM coefficient <lhs|L|rhs>*i
    Returns tuple (Lx, Ly, Lz)
    """

    orbinput = integration._standardize_orbital_input(
            lhs.origin, rhs.origin,
            lhs.orientation, rhs.orientation,
            lhs.type_, rhs.type_)

    p1arr, p2arr, r12 = orbinput

    l = np.zeros(3)

    for a1,d1 in lhs.data:
        for a2,d2 in rhs.data:

            drvset1 = []
            drvset2 = []

            for k in range(3):
                drvset1.append(integration._drv_1d(a1, p1arr, k))
                drvset2.append(integration._drv_1d(a2, p2arr, k))

            for k in range(3):

                for c1, drv1 in drvset1[k-2]:
                    for c2, drv2 in drvset2[k-1]:
                        for rc, z in zip(atom_coords, atom_charges):
                            l[k] += z*c1*c2*d1*d2*integration._nu_coul_3d(a1, a2, rc-lhs.origin, drv1, drv2, r12)

                for c1, drv1 in drvset1[k-1]:
                    for c2, drv2 in drvset2[k-2]:
                        for rc, z in zip(atom_coords, atom_charges):
                            l[k] -= z*c1*c2*d1*d2*integration._nu_coul_3d(a1, a2, rc-lhs.origin, drv1, drv2, r12)


    return l
