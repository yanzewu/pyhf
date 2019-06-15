
def analyze_rci(ci_datasets, C, bases):
    
    print('\nConfiguration Interaction Analysis\n')
    for degen, E, V, mo_orbitals in ci_datasets:
        print('%s energies:' % ('singlet' if degen == 's' else 'triplet'))
        print('\n'.join(('%.4g' % E_ for E_ in E)))
            