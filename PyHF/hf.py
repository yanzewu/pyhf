
import json
import numpy as np

from . import basis
from . import integration
from . import roothaan
from . import sprint

_default_printer = sprint.SPrinter()


def create_hf_matrices(bases, nuclear_coords, nuclear_charges):
    """ Evaluating integrals for list of basis.
    Args:
        bases: list of basis.Basis object.
    Returns:
        S: overlap matrix;  S_ij = (i|j)
        h: kernel matrix;   h_ij = (i|h|j)
        v: electron Coulomb matrix; v_ijkl = (ij|kl)
    """

    n_bases = len(bases)

    A = np.zeros(n_bases)   # normalization
    S = np.zeros((n_bases, n_bases))
    h = np.zeros((n_bases, n_bases))
    v = np.zeros((n_bases, n_bases, n_bases, n_bases))

    for i in range(n_bases):
        A[i] = np.sqrt(integration.evaluate_overlap(bases[i], bases[i]))

    for i in range(n_bases):
        S[i, i] = 1
        for j in range(i):
            S[i, j] = integration.evaluate_overlap(bases[i], bases[j])/(A[i]*A[j])
            S[j, i] = S[i, j]

    for i in range(n_bases):
        for j in range(i + 1):
            h[i, j] = integration.evaluate_kernel(bases[i], bases[j], nuclear_coords, nuclear_charges)/(A[i]*A[j])
            h[j, i] = h[i, j]

    for i in range(n_bases):
        for j in range(i + 1):
            ij = i*(i+1)/2+j
            for k in range(n_bases):
                for l in range(k + 1):
                    if ij > k*(k+1)/2+l:
                        continue
                    if v[i,j,k,l] != 0:
                        continue
                    v[i,j,k,l] = integration.evaluate_2electron_coulomb(bases[i], bases[j], bases[k], bases[l])/(A[i]*A[j]*A[k]*A[l])
                    v[j,i,l,k] = v[j,i,k,l] = v[i,j,l,k] = v[i,j,k,l]
                    v[l,k,j,i] = v[k,l,j,i] = v[l,k,i,j] = v[k,l,i,j] = v[i,j,k,l]

    return S, h, v


def rhf(atom_charges, atom_coords, net_charge=0, basis_set='sto-3g', C_init=None, n_step=500, atol=1e-7, rtol=1e-5, mr=0.5, printer=_default_printer):
    """ 
    Performing restricted Hatree Fock Calculation (including single-electron case).
    
    Args:
        atom_names: list of atom names;
        atom_coords: list of array;
        net_charge: net charge of the molecule;
        basis_set: name of basis set;
        C_init: initial guess of state matrix;
        n_step: maximum step;
        atol, rtol: controls convergence;
        mr: initial mixing rate of fock matrix;
        printer: sprint.SPrinter instance;
    Returns:
        E: array of energy;
        E_core: array of core energy
        C: orbital matrix;
        S: overlap matrix;
        n_orbital: number of orbitals
        bases: list of Basis object
    """

    printer.warning('Performing restricted Hartree-Fock calculation\n')

    printer.debug('Nuclear positions:')
    printer.debug('\n'.join('%d\t%s'%(c,d) for c,d in zip(atom_charges, atom_coords)))

    n_electron = sum(atom_charges) - net_charge
    assert n_electron > 0, 'No electron'
    assert n_electron == 1 or n_electron % 2 == 0, 'Electron number must be even or 1'

    n_orbital = n_electron//2 if n_electron > 1 else 1

    printer.info('Total %d atoms, %d orbitals. Assigning basis' % (len(atom_charges), n_orbital))
    bases = basis.construct_basis(basis_set, atom_charges, atom_coords)

    printer.info('Total %d bases assigned. Calculating matrices' % len(bases))
    S, h, v = create_hf_matrices(bases, atom_coords, atom_charges)

    printer.info('SCF iteration start\n')
    if C_init is None:
        E, C = roothaan.diagonal(h, S)  # initial guess
    else:
        C = C_init.copy()
        E = np.zeros(len(bases))

    F = roothaan.build_fock(C, h, v, n_orbital)

    E_last = E
    mr0 = mr

    for n in range(n_step):

        if n_electron == 1:
            printer.warning('Only 1 electron: Skip SCF')
            break

        C_old = C.copy()
        E, C = roothaan.diagonal(F, S)

        printer.debug('[Step %d]\t%.6e' % (n+1, np.sum(E[:n_orbital])))

        if np.sum(np.abs(E - E_last)) > atol + rtol*np.sum(np.abs(E)):
            mr = mr0 * 0.5**(n//5)
            F = F*mr + roothaan.build_fock(C, h, v, n_orbital)*(1-mr)
            E_last = E
        elif mr > 0:
            mr = 0
            F = roothaan.build_fock(C, h, v, n_orbital)
            E_last = E
        else:
            break

    if n == n_step-1:
        printer.warning('Maximum step (%d) reached, energy not converged' % n)
    else:
        printer.warning('Energy converged')

    return E, C, S, h, v, n_orbital, bases


def uhf(atom_charges, atom_coords, net_charge, n_single_electron, basis_set='sto-3g', C_init=None, n_step=500, atol=1e-7, rtol=1e-5, mr=0.5, printer=_default_printer):
    """ Performing unrestricted Hartree-Fock calculation.
    Args:
        atom_charges: list of atom charges;
        atom_coords: list of array objects;
        net_charge: net charge of the molecule;
        n_single_electron: number of single electrons;
        basis_set: name of basis set;
        C_init: tuple (C_alpha, C_beta), initial guess of state matrix;
        n_step: Maximum calculation step;
        atol, rtol: controls convergence;
        mr: initial mixing rate of fock matrix;
        printer: sprint.SPrinter instance;
    Returns:
        E: array of energy, in (alpha, beta) pair
        E_core: array of core energy, in (alpha, beta) pair
        C: orbital matrix, in (alpha, beta) pair
        S: overlap matrix;
        n_orbital: number of orbitals, in (alpha, beta) pair
        bases: list of basis.Basis objects
    """

    printer.warning('Performing unrestricted Hartree-Fock calculation\n')

    printer.debug('Nuclear positions:')
    printer.debug('\n'.join('%d\t%s'%(c,d) for c,d in zip(atom_charges, atom_coords)))

    n_electron = np.sum(atom_charges) - net_charge
    n_alpha = (n_electron + n_single_electron)//2
    n_beta = (n_electron - n_single_electron)//2

    assert n_alpha >= 0 and n_beta >= 0, 'No electron'

    printer.info('Total %d atoms, %d alpha orbitals, %d beta orbitals. Assigning basis' % (len(atom_charges), n_alpha, n_beta))
    bases = basis.construct_basis(basis_set, atom_charges, atom_coords)

    printer.info('Total %d bases assigned. Calculating matrices' % len(bases))
    S, h, v = create_hf_matrices(bases, atom_coords, atom_charges)

    printer.info('SCF iteration start\n')
    if C_init is None:
        E_a, C_a = roothaan.diagonal(h, S)  # initial guess
        C_b = C_a.copy()
    else:
        E_a = np.zeros(len(bases))
        C_a = C_init[0].copy()
        C_b = C_init[1].copy()

    F_a, F_b = roothaan.build_fock_u((C_a, C_b), h, v, (n_alpha, n_beta))

    E_a_last = E_a
    E_b_last = E_a
    mr0 = mr

    for n in range(n_step):

        E_a, C_a = roothaan.diagonal(F_a, S)
        E_b, C_b = roothaan.diagonal(F_b, S)

        printer.debug('[Step %d]\t%.6e' % (n+1, sum(E_a[:n_alpha]) + sum(E_b[:n_alpha])))

        if abs(np.sum(E_a - E_a_last) + np.sum(E_b - E_b_last)) > atol + rtol*abs(np.sum(E_a) + np.sum(E_b)):
            mr = mr0 * 0.5**(n//5)
            F_a_1, F_b_1 = roothaan.build_fock_u((C_a, C_b), h, v, (n_alpha, n_beta))
            F_a = F_a*mr + F_a_1*(1-mr)
            F_b = F_b*mr + F_b_1*(1-mr)
            E_a_last = E_a
            E_b_last = E_b
        elif mr > 0:
            mr = 0
            F_a, F_b = roothaan.build_fock_u((C_a, C_b), h, v, (n_alpha, n_beta))
            E_a_last, E_b_last = E_a, E_b
        else:
            break


    if n == n_step-1:
        printer.warning('Maximum step (%d) reached, energy not converged' % n)
    else:
        printer.warning('Energy converged')

    return (E_a, E_b), (C_a, C_b), S, h, v, (n_alpha, n_beta), bases


def rohf(atom_charges, atom_coords, net_charge, n_single_electron, basis_set='sto-3g', C_init=None, n_step=500, \
        atol=1e-7, rtol=1e-5, mr=0.5, mix_style='guest', printer=_default_printer):
    """ Performing restricted open shell Hartree-Fock calculation.
    Args:
        atom_charges: list of atom charges;
        atom_coords: list of array objects;
        net_charge: net charge of the molecule;
        n_single_electron: number of single electrons;
        basis_set: name of basis set;
        C_init: tuple (C_alpha, C_beta), initial guess of state matrix;
        n_step: Maximum calculation step;
        atol, rtol: controls convergence;
        mr: initial mixing rate of fock matrix;
        printer: sprint.SPrinter instance;
    Returns:
        E: array of energy, in (alpha, beta) pair
        E_core: array of core energy, in (alpha, beta) pair
        C: orbital matrix, in (alpha, beta) pair
        S: overlap matrix;
        n_orbital: number of orbitals, in (alpha, beta) pair
        bases: list of basis.Basis objects
    """

    printer.warning('Performing restricted open shell Hartree-Fock Calculation\n')

    printer.debug('Nuclear positions:')
    printer.debug('\n'.join('%d\t%s'%(c,d) for c,d in zip(atom_charges, atom_coords)))

    n_electron = np.sum(atom_charges) - net_charge
    n_alpha = (n_electron + n_single_electron)//2
    n_beta = (n_electron - n_single_electron)//2

    assert n_alpha >= 0 and n_beta >= 0, 'No electron'
    assert n_alpha >= n_beta

    mix_coeff = {
        'davidson':(0.5, 1.0, 1.0, 0.0),
        'guest':(0.5, 0.5, 0.5, 0.5),
        'roothaan':(-0.5, 1.5, 0.5, 0.5),
        'gvb':(0.5, 0.5, 0.5, 0.0),
        'canonical':(0.0, 1.0, 1.0, 0.0)
    }[mix_style]

    printer.info('Total %d atoms, %d alpha orbitals, %d beta orbitals. Assigning basis' % (len(atom_charges), n_alpha, n_beta))
    bases = basis.construct_basis(basis_set, atom_charges, atom_coords)

    printer.info('Total %d bases assigned. Calculating matrices' % len(bases))
    S, h, v = create_hf_matrices(bases, atom_coords, atom_charges)

    printer.info('SCF iteration start\n')
    if C_init is None:
        E, C = roothaan.diagonal(h, S)  # initial guess
    else:
        E = np.zeros(len(bases))
        C = C_init

    F_a, F_b = roothaan.build_fock_u((C, C), h, v, (n_alpha, n_beta))
    F = roothaan.fock2ro(F_a, F_b, n_alpha, n_beta, *mix_coeff)
    E_last = E
    mr0 = mr

    for n in range(n_step):

        E, C = roothaan.diagonal(F, S)

        printer.debug('[Step %d]\t%.6e' % (n+1, sum(E[:n_alpha])))

        if abs(np.sum(E - E_last)) > atol + rtol*abs(np.sum(E)):
            mr = mr0 * 0.5**(n//5)
            F_a, F_b = roothaan.build_fock_u((C, C), h, v, (n_alpha, n_beta))
            F = F*mr + roothaan.fock2ro(F_a, F_b, n_alpha, n_beta, *mix_coeff)*(1-mr)
            E_last = E
        elif mr > 0:
            mr = 0
            F_a, F_b = roothaan.build_fock_u((C, C), h, v, (n_alpha, n_beta))
            F = roothaan.fock2ro(F_a, F_b, n_alpha, n_beta, *mix_coeff)
            E_last = E
        else:
            break


    if n == n_step-1:
        printer.warning('Maximum step (%d) reached, energy not converged' % n)
    else:
        printer.warning('Energy converged')

    return (roothaan.diagonal(F_a, S)[0], roothaan.diagonal(F_b, S)[0]), (C, C), S, h, v, (n_alpha, n_beta), bases