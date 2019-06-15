
def analyze_rci(ci_datasets, C, bases):
    
    for degen, E, V, mo_orbitals in ci_datasets:
        print('%s energies:' % ('singlet' if degen == 1 else 'triplet'))
        print('\n'.join(('%.4g' % E_ for E_ in E)))
            