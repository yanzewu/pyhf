
import numpy as np

from . import postproc
from . import visualize


def spin_stat(coeff, mo_orbitals):
    """ Identify spin of a orbital combination.
    """
    spin_coeff = np.zeros(4)
    ave_ms = 0.0
    for k in range(len(coeff)):
        spin_coeff[mo_orbitals[k].degen-1] += np.abs(coeff[k])**2
        ave_ms += mo_orbitals[k].ms * np.abs(coeff[k])**2

    return spin_coeff, ave_ms


def analyze_rci(ci_datasets, C, bases, E_nu):
    
    print('\nConfiguration Interaction Analysis\n')

    for d in ci_datasets:
        if d[0] == 's' or d[0] is None:
            E0 = min(d[1])

    # Labeling data

    orbital_data = []

    for degen, E, V, H, mo_orbitals in ci_datasets:
        for i, E_ in enumerate(E):
            orbital_data.append((E_, V[:,i], mo_orbitals, '%s%d'%(degen.upper() if degen else '', i if degen == 's' or degen is None else i+1)))

    orbital_data.sort(key=lambda x:x[0])

    if ci_datasets[0][0] is not None:

        print('Name\tE(Abs.)/Ha\tE(Rel.)/eV')
        for E_, V_, mo_, name_ in orbital_data:
            print('%s\t%6g\t%6g' % (name_, E_+E_nu, (E_-E0)*27.2114))

    else:   # Spin analysis is needed

        print('Name\tE(Abs.)/Ha\tE(Rel.)/eV\tSpin')
        for E_, V_, mo_, name_ in orbital_data:

            spin_coeff, ave_ms = spin_stat(V_, mo_)
            spin_coeff_w_degen = sorted(zip(spin_coeff, ('S','D','T','Q')), key=lambda d:d[0], reverse=True)

            spin_str = ''
            for c, d in spin_coeff_w_degen:
                if 1 - c < 5e-5:
                    spin_str = d + '+'
                    break
                elif c < 5e-5:
                    break
                else:
                    spin_str += '%.4f%s+' % (c, d)
            spin_str = spin_str[:-1] + '(ms=%.2g)' % (ave_ms if abs(ave_ms) > 5e-5 else 0)

            print('%s\t%6g\t%6g\t%s' % (name_, E_+E_nu, (E_-E0)*27.2114, spin_str))
