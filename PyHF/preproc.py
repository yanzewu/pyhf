
import numpy as np
import json

from . import sprint

def _getcwd():

    import sys
    import os
    return os.path.split(sys.argv[0])[0]


def read_input(fp):
    """ Generate input for Hartree fock functions.
    Args:
        fp: file-like object;
    Returns:
        name: name of the session;
        hftype: 'rhf'/'uhf';
        hf_kwargs: args passed to HF function rhf()/uhf();
        post_opts: options for analyze_hf();
        scan_opts: options for scanning;
    """

    data = json.load(fp)

    with open('%s/atom.json'%_getcwd(), 'r') as atom_fp:
        atom_data = json.load(atom_fp)    

    # HF args
    atom_charges = [atom_data[a]['charge'] for a in data['atoms']]

    hf_kwargs = {
        'basis_set': data.get('basis_set', 'sto-3g'),
        'n_step': data.get('n_step', 500),
        'atol': data.get('atol', 1e-9),
        'rtol': data.get('rtol', 1e-7),
        'net_charge':data.get('charge', 0),
        'atom_coords':np.array(data['coords']),
        'atom_charges': atom_charges
    }

    if 'hftype' not in data:
        total_charge = sum(atom_charges) - hf_kwargs['net_charge']    
        hftype = 'uhf' if total_charge > 1 and total_charge % 2 == 1 else 'rhf'
    else:
        hftype = data['hftype']

    try:
        if hftype == 'uhf':
            hf_kwargs['n_single_electron'] = data['n_single_electron']
    except KeyError:
        print('Please specify single electron number for UHF')
        raise

    # Verbose
    if 'verbose' in data:
        str2level = {'full':sprint.FULL,'normal':sprint.NORMAL,'minimal':sprint.MINIMAL,'silent':sprint.SILENT}
        sprint.SPrinter.global_level = str2level[data['verbose']]

    # Post analysis
    post_opts = data.get('post_analysis', [])
    assert isinstance(post_opts, list)

    # Generating scanning options
    scan_kwargs = None
    if 'scan' in data and data['scan'].get('enable', True) == True:
       
        repl_coord = []
        repl_expr = []
        var_list = {}

        for n, v in data['scan'].items():
            if n[0].isdigit():
                repl_coord.append([int(n.split(',')[0])-1,int(n.split(',')[1])-1])
                repl_expr.append(v)
            else:
                assert isinstance(v, list)
                assert len(v)==3
                var_list[n] = v

        scan_kwargs = {
            'repl_coord':repl_coord, 
            'repl_expr':repl_expr, 
            'var_list':var_list
        }


    return data.get('name', 'HFProject1'), hftype, hf_kwargs, post_opts, scan_kwargs


def create_scan_coords(coord0, repl_coord, repl_expr, var_list):
    """ Creating list of scanning coordination.
    Args:
        coord0: nx3 array object;
        repl_coord: nx2 array object, locations to be replaced;
        repl_expr: list of string, can be executed by 'eval'
        vars: dict(var=[start,stop,step]), stop is taken.

    Returns:
        list of coords.
    """

    npfunc = {
        "sin":np.sin,
        "cos":np.cos,
        "tan":np.tan,
        "exp":np.exp,
        "ln":np.log,
        "mul":np.multiply,
        "div":np.divide,
        "sum":np.sum,
        "abs":np.abs,
        "sqrt":np.sqrt,
        "pi":np.pi
    }

    var_values = dict(((v, np.arange(s[0],s[1]+s[2]/2,s[2])) for v,s in var_list.items()))

    repl_values = [eval(expr, var_values.copy(), npfunc) for expr in repl_expr]

    n_steps = len(repl_values[0])

    coord_list = [coord0.copy() for i in range(n_steps)]

    for i in range(n_steps):
        for j, x in zip(repl_coord, repl_values):
            coord_list[i][tuple(j)] = x[i]

    return coord_list, var_values