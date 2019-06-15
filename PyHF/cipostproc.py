

from . import postproc
from . import visualize

def analyze_rci(ci_datasets, C, bases, E_nu):
    
    print('\nConfiguration Interaction Analysis\n')

    for d in ci_datasets:
        if d[0] == 's':
            E0 = min(d[1])

    # Labeling data

    orbital_data = []

    for degen, E, V, mo_orbitals in ci_datasets:
        for i, (E_, V_, mo_) in enumerate(zip(E, V, mo_orbitals)):
            orbital_data.append((E_, V_, mo_, '%s%d'%(degen, i if degen == 's' else i+1)))

    orbital_data.sort(key=lambda x:x[0])

    print('Name\tE(Abs.)/Ha\tE(Rel.)/eV')
    for d in orbital_data:
        print('%s\t%6g\t%6g' % (d[3], d[0]+E_nu, (d[0]-E0)*27.2114))
