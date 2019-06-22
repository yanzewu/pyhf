"""Sub functions and Main function of Post processing
"""

import numpy as np
from . import roothaan
from . import visualize
from . import rci
from . import cipostproc


def charge2name(c):
    return ['H','He','Li','Be','B','C','N','O','F'][c-1]

def nuclear_energy(atom_coords, atom_charges):
    E_nu = 0.0

    for i in range(len(atom_coords)):
        for j in range(i):
            E_nu += atom_charges[i]*atom_charges[j]/np.linalg.norm(atom_coords[i]-atom_coords[j])

    return E_nu


def density_matrix_general(C, orbitals):
    return C[:,orbitals].dot(C[:,orbitals].T)

def density_matrix(C, n_orbital):
    """ Returns the density matrix D of state matrix. Simply a wrapper of
        roothann.build_density_mat()
    """

    if type(C) is tuple:
        return density_matrix_general(C[0], range(n_orbital[0])), density_matrix_general(C[1], range(n_orbital[1]))
    else:
        return 2*density_matrix_general(C, range(n_orbital))

def eff_spin(C_a, C_b, n_a, n_b, S):
    """ Return <S^2> for UHF.
    """
    overlap_mat = C_a[:,:n_a].T.dot(S.dot(C_b[:,:n_b]))
    tot_overlap = np.linalg.norm(overlap_mat)**2
    return (n_a-n_b)*(n_a-n_b+2)/4 + n_b - tot_overlap


def muliken(P, atom_charges):
    """ Return the muliken charge of atoms.
    Args:
        P: Real density matrix (D*S);
        atom_charges: list of atom charges.
    Returns:
        array of charge of each atom.
    """

    gop = np.sum(P, axis=0)

    atom_of_orbital = []
    for i, c in enumerate(atom_charges):
        if c < 3:
            atom_of_orbital += [i]
        else:
            atom_of_orbital += [i,i,i,i,i]
    # notice the algorithm need to be updated when atoms in period 3 are included

    assert len(atom_of_orbital) == P.shape[0]

    charge_density = np.zeros(len(atom_charges))
    for g, a in zip(gop, atom_of_orbital):
        charge_density[a] += g

    return np.array(atom_charges) - charge_density


def orbital_analysis(restricted, n_a, n_b, C_a, C_b, E_a, E_b, S, h, atom_coords, atom_charges, options):
    """ Performing routine analysis of HF result.
    """

    # core energy
    E_core_a = np.array([C_a[:,i].T.dot(h.dot(C_a[:,i])) for i in range(len(C_a))])
    E_core_b = np.array([C_b[:,i].T.dot(h.dot(C_b[:,i])) for i in range(len(C_b))])

    E_core_tot = np.sum(E_core_a[:n_a]) + np.sum(E_core_b[:n_b])
    E_orb_tot = np.sum(E_a[:n_a]) + np.sum(E_b[:n_b])
    E_nu = nuclear_energy(atom_coords, atom_charges)

    print('Total Electronic Energy =\t%.10g' % ((E_orb_tot + E_core_tot)/2))
    print('Electronic Core Energy=\t%.10g' % E_core_tot)
    print('Electronic Columb Energy=\t%.10g' % ((E_orb_tot - E_core_tot)/2))
    print('Nuclear Energy =\t%.10g' % E_nu)
    print('Total Energy =\t%.10g' % ((E_orb_tot + E_core_tot)/2 + E_nu))

    if 'orbital-energy' in options:

        print('\nOrbital energies:')
        print('Index (Status)\tEnergy/Ha')

        if restricted:
            for i in range(len(C_a)):
                print('%d (%s)\t%g' % (
                    i+1, 'occupied' if i < n_a else 'virtual',
                    E_a[i]))
        else:
            occupicy = np.zeros(len(C_a)*2, dtype=int)
            occupicy[:n_a] = 1
            occupicy[len(C_a):len(C_a)+n_b] = 1
            spin = np.zeros(len(C_a)*2)
            spin[:len(C_a)] = 1

            E = np.concatenate((E_a, E_b))
            idx = np.argsort(E)
            for i in idx:
                print('%d%s (%s)\t%g' % (
                    i+1 if spin[i] == 1 else i-len(C_a)+1,
                    'a' if spin[i] == 1 else 'b',
                    'occupied' if occupicy[i] == 1 else 'virtual',
                    E[i]))

    if 'charge-muliken' in options or 'muliken' in options:

        D = (density_matrix(C_a, n_a) + density_matrix(C_b, n_b))/2

        print('\nMuliken charge analysis:')
        print('Atom\tCharge(Muliken)')
        print('\n'.join(('%d %s\t%.4g'%(i, charge2name(atom_charges[i]), c) for i, c in enumerate(muliken(D*S, atom_charges)))))

    return (E_orb_tot + E_core_tot)/2, E_nu


def analyze_hf(hftype, *args, **kwargs):
    if hftype == 'rhf':
        return analyze_rhf(*args, **kwargs)
    elif hftype == 'uhf':
        return analyze_uhf(*args, **kwargs)
    elif hftype == 'rohf':
        return analyze_uhf(*args, **kwargs)
    else:
        raise ValueError(hftype)


def analyze_rhf(E, C, S, h, v, n_orbital, bases, atom_coords, atom_charges, name='', options={}):
    """ Analyze and print the result of RHF.
    Args:
        E, C, S, h, v, n_orbital, bases: The output of rhf();
        atom_coords: Nx3 array, coordination of atoms;
        atom_charges: list of atom charges;
        name: name of the system, used in plotting;
        options: optional analysis. See the document for the full list.
    """

    E_ele, E_nu = orbital_analysis(True, n_orbital, n_orbital, C, C, E, E, S, h, atom_coords, atom_charges, options)

    if 'plot' in options:
        visualize.ui_plot('rhf', C, bases, n_orbital, atom_charges, atom_coords, name)

    if 'mp2' in options:
        from . import mopt
        Ecorr = mopt.rmp2_energy(n_orbital, C, v, E)
        print('\nMP2 Correlation Energy =\t%g' % Ecorr)
        print('MP2 Total Energy =\t%g' % (E_ele + E_nu + Ecorr))

    if 'ci' in options and options['ci']:
        
        cioutput = rci.rci(n_orbital, C, S, h, v, **options.get('ci_kwargs', {}))
        cipostproc.analyze_rci(cioutput, C, bases, E_nu)

    if 'cis-soc' in options and options['cis-soc']:
        from . import soc
        cioutput = soc.rcis_with_soc(n_orbital, C, S, h, v, bases, atom_coords, atom_charges, **options.get('cis-soc_kwargs', {}))
        cipostproc.analyze_rci(cioutput, C, bases, E_nu)


def analyze_uhf(E, C, S, h, v, n_orbital, bases, atom_coords, atom_charges, name='', options=[]):
    """Analyze and print the result of RHF.
    Args:
        E, C, S, h, v, n_orbital, bases: The output of uhf();
        atom_coords: Nx3 array, coordination of atoms;
        atom_charges: list of atom charges;
        name: name of the system, used in plotting;
        options: optional analysis. See the document for the full list.
    """

    orbital_analysis(False, n_orbital[0], n_orbital[1], C[0], C[1], E[0], E[1], S, h, atom_coords, atom_charges, options)
    
    if C[0] is not C[1]:
        tot_spin = eff_spin(C[0], C[1], n_orbital[0], n_orbital[1], S)
        print('\n<S^2>: %.4f' % tot_spin)
            
    if 'plot' in options:
        visualize.ui_plot('uhf', C, bases, n_orbital, atom_charges, atom_coords, name)
    
