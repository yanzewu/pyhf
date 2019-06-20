"""Sub functions and Main function of Post processing
"""

import numpy as np
from . import roothaan
from . import visualize
from . import rci
from . import cipostproc


def nuclear_energy(atom_coords, atom_charges):
    E_nu = 0.0

    for i in range(len(atom_coords)):
        for j in range(i):
            E_nu += atom_charges[i]*atom_charges[j]/np.linalg.norm(atom_coords[i]-atom_coords[j])

    return E_nu


def total_energy_rhf(E, E_core, n_orbital, atom_coords, atom_charges):

    return np.sum((E+E_core)[:n_orbital]) + nuclear_energy(atom_coords, atom_charges)


def total_energy_uhf(E, E_core, n, atom_coords, atom_charges):

    return (np.sum((E[0]+E_core[0])[:n[0]]) + np.sum((E[1]+E_core[1])[:n[1]]))/2 + nuclear_energy(atom_coords, atom_charges)


def density_matrix_general(C, orbitals):
    D = np.zeros_like(C)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            D[i, j] = sum([C[i, n]*C[j, n] for n in orbitals])

    return D

def density_matrix(C, n_orbital):
    """ Returns the density matrix D of state matrix. Simply a wrapper of
        roothann.build_density_mat()
    """

    if type(C) is tuple:
        return density_matrix_general(C[0], range(n_orbital[0])), build_density_mat(C[1], range(n_orbital[1]))
    else:
        return density_matrix_general(C, range(n_orbital))


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


def analyze_hf(hftype, *args, **kwargs):
    if hftype == 'rhf':
        return analyze_rhf(*args, **kwargs)
    elif hftype == 'uhf':
        return analyze_uhf(*args, **kwargs)
    else:
        raise ValueError(hftype)


def analyze_rhf(E, E_core, C, S, h, v, n_orbital, bases, atom_coords, atom_charges, name='', options={}):
    """ Analyze and print the result of RHF.
    Args:
        E, E_core, C, S, n_orbital, bases: The output of rhf();
        atom_coords: Nx3 array, coordination of atoms;
        atom_charges: list of atom charges;
        name: name of the system, used in plotting;
        options: optional analysis. Available options are:
            orbital-energy
            muliken-charge
            density-matrix
            plot
    """

    E_tot = E + E_core  # Total energy of each orbital
    E_e2 = E - E_core    # Electron-electron energy of each orbital
    E_ele = np.sum((E+E_core)[:n_orbital])  # Total electronic energy
    E_nu = nuclear_energy(atom_coords, atom_charges)

    D = density_matrix(C, n_orbital)

    print('Total Electronic Energy =\t%.10g' % E_ele)
    print('Core Energy=\t%.10g' % np.sum(E_core[:n_orbital]*2))
    print('2-electron Energy=\t%.10g' % np.sum(E_e2[:n_orbital]))
    print('Nuclear Energy =\t%.10g' % E_nu)
    print('Total Energy =\t%.10g' % (E_ele + E_nu))


    if 'orbital-energy' in options:
        # Energy here represents energy of the orbital (2 electrons)

        print('\nOrbital energies:')
        print('Index (Status)\tOrbital\tCore\t2Electron\tTotal')
        for i in range(len(bases)):
            print('%d (%s)\t%g\t%g\t%g\t%g' % (
                i+1, 'occupied' if i < n_orbital else 'virtual',
                E[i],
                E_core[i]*2,
                E_e2[i],
                E_tot[i]
                ))

    if 'density-matrix' in options:
        P = D*S
        print('\nDensity matrix:')
        print('\n'.join(['\t'.join(
            '%.4e'% P[i,j] for j in range(D.shape[1])
            ) for i in range(D.shape[0])]))

    if 'charge-muliken' in options:
        print('\nAtom\tCharge(Muliken)')
        print('\n'.join(('%d\t%.4g'%(i, c) for i, c in enumerate(muliken(D*S, atom_charges)))))

    if 'plot' in options:
        visualize.ui_plot('rhf', C, bases, n_orbital, atom_charges, atom_coords, name)

    if 'ci' in options and options['ci']:
        
        cioutput = rci.rci(n_orbital, C, S, h, v, **options.get('ci_kwargs', {}))
        cipostproc.analyze_rci(cioutput, C, bases, E_nu)

    if 'cis-soc' in options and options['cis-soc']:
        from . import soc
        cioutput = soc.rcis_with_soc(n_orbital, C, S, h, v, bases, atom_coords, atom_charges, **options.get('cis-soc_kwargs', {}))
        cipostproc.analyze_rci(cioutput, C, bases, E_nu)


def analyze_uhf(E, E_core, C, S, n_orbital, bases, atom_coords, atom_charges, name='', options=[]):
    """Analyze and print the result of RHF.
    Args:
        E, E_core, C, S, n_orbital, bases: The output of uhf();
        atom_coords: Nx3 array, coordination of atoms;
        atom_charges: list of atom charges;
        name: name of the system, used in plotting;
        options: optional analysis. Available options are:
            orbital-energy
            muliken-charge
            density-matrix
            plot
    """

    n_a, n_b = n_orbital

    E_ele = (np.sum((E[0] + E_core[0])[:n_a]) + np.sum((E[1] + E_core[1])[:n_b]))/2
    E_nu = nuclear_energy(atom_coords, atom_charges)

    D_a, D_b = density_matrix(C, n_orbital)

    # TODO: Check summations
    print('Total Electronic Energy =\t%.10g' % E_ele)
    print('Core Energy=\t%.10g' % np.sum(E_core))
    print('2-electron Energy=\t%.10g' % (E_ele - np.sum(E_core)))
    print('Nuclear Energy =\t%.10g' % E_nu)
    print('Total Energy =\t%.10g' % (E_ele + E_nu))

    if 'orbital-energy' in options:
        # Summation of orbital energies here will be 2*E_ele

        E = np.concatenate((E[0], E[1]))
        E_core = np.concatenate((E_core[0], E_core[1]))
        E_tot = E + E_core
        E_2e = E - E_core
        spin = np.zeros(len(bases)*2)
        spin[:len(bases)] = 1
        occupicy = np.zeros(len(bases)*2)
        occupicy[:n_a] = 1
        occupicy[len(bases):len(bases)+n_b] = 1

        idx = np.argsort(E)

        print('\nOrbital energies:')
        print('Index (Status)\tOrbital\tCore\t2Electron\tTotal')
        for j, i in enumerate(idx):
            print('%d%s (%s)\t%g\t%g\t%g\t%g' % (
                i+1 if spin[i] == 1 else i-len(bases)+1,
                'a' if spin[i] == 1 else 'b',
                'occupied' if occupicy[i] == 1 else 'virtual',
                E[i],
                E_core[i],
                E_2e[i],
                E_tot[i]
                ))

    if 'density-matrix' in options:

        print('\nDensity matrix: Alpha')
        print('\n'.join(['\t'.join(
            '%.4e'% D_a[i,j] for j in range(D_a.shape[1])
            ) for i in range(D_a.shape[0])]))

        print('\nDensity matrix: Beta')
        print('\n'.join(['\t'.join(
            '%.4e'% D_b[i,j] for j in range(D_b.shape[1])
            ) for i in range(D_b.shape[0])]))

    if 'charge-muliken' in options:
        print('\nAtom\tCharge(Muliken)')
        print('\n'.join(('%d\t%.4g'%(i, c) for i, c in enumerate(muliken((D_a + D_b)*S, atom_charges)))))
            
    if 'plot' in options:
        visualize.ui_plot('rhf', C, bases, n_orbital, atom_charges, atom_coords, name)
    