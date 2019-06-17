

from . import postproc
from . import visualize

def analyze_rci(ci_datasets, C, bases, E_nu):
    
    print('\nConfiguration Interaction Analysis\n')

    for d in ci_datasets:
        if d[0] == 's' or d[0] is None:
            E0 = min(d[1])

    # Labeling data

    orbital_data = []

    for degen, E, V, H, mo_orbitals in ci_datasets:
        for i, (E_, V_, mo_) in enumerate(zip(E, V, mo_orbitals)):
            orbital_data.append((E_, V_, mo_, '%s%d'%(degen.upper() if degen else '', i if degen == 's' or degen is None else i+1)))

    orbital_data.sort(key=lambda x:x[0])

    print('Name\tE(Abs.)/Ha\tE(Rel.)/eV')
    for E_, V_, mo_, name_ in orbital_data:
        print('%s\t%6g\t%6g' % (name_, E_+E_nu, (E_-E0)*27.2114))
