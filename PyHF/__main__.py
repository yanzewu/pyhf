
import sys

from . import preproc
from . import hf
from . import postproc
from . import visualize

if len(sys.argv) > 1:

    try:
        with open(sys.argv[1], 'r') as input_fp:
            name, hftype, kwargs, post_opts, scan_kwargs = preproc.read_input(input_fp)
    except OSError:
        print('Cannot open "%s"' % sys.argv[1])
        raise
    
    if scan_kwargs is None:

        if hftype == 'uhf':
            ret = hf.uhf(**kwargs)
        elif hftype == 'rhf':
            ret = hf.rhf(**kwargs)
        elif hftype == 'rohf':
            ret = hf.rohf(**kwargs)
        
        postproc.analyze_hf(hftype, *ret, kwargs['atom_coords'], kwargs['atom_charges'], name=name, options=post_opts)
        
    else:
        coords_list, var_values = preproc.create_scan_coords(kwargs['atom_coords'], **scan_kwargs)
        
        print('Scanning... Total %d steps' % len(coords_list))

        for i, c in enumerate(coords_list):
            print('\n[Scan Step %d]' % i)
            print('\t'.join(["%s=%g"%(k,v[i]) for k,v in var_values.items()]))

            kwargs['atom_coords'] = c
           
            if hftype == 'uhf':
                ret = hf.uhf(**kwargs)
            elif hftype == 'rhf':
                ret = hf.rhf(**kwargs)
                    
            postproc.analyze_hf(hftype, *ret, kwargs['atom_coords'], kwargs['atom_charges'], name=name, options=post_opts)

else:
    print('python -m PyHF [input_file]')