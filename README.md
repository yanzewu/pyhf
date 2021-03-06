# PyHF

Python Hartree-Fock (RHF, ROHF and UHF) and configuration interaction implementation.

## Usage

    python -m PyHF [input.json]

A example input is shown in [h2.json](example/h2.json). General specification of json is

- name;
- hftype (optional): rhf/rohf/uhf;
- basis_set: currently only sto-3g;
- atoms: list of atoms (currently only H to F);
- coords: atom coords, Nx3 array;
- charge (optional): net charge of molecule;
- n_single_electron: required in ROHF and UHF;
- n_step (optional);
- post_analysis (optional): available options:
    - orbital-energy
    - density-matrix
    - charge-muliken
    - plot: If plot is specified, a prompt will appear after calculation, which accepts
        * "charge"
        * number representing orbital: 1,2,3,... in RHF, 1a,2b,... in UHF; Can be seperated by space.
        * "homo": will show number of HOMO
        * "q": exit
    - mp2: Calculating MP2 energy. Currently only for closed shell.
    - ci: Performing configuration interaction (CI) calculation, compatible with scan. Currently only single-excitation with closed shell system (RCIS) is available. To perform CI, post_analysis must be in _object format_ (e.g. "{key:data}"), see [hf-ci.json](example/hf-ci.json) as example.
    - ci_kwargs: Keyword argument passed to CI.
        * level: 's'/'d'/'sd', only 's' (default) is avaiable
        * degeneracy: 's'/'t'/'st' (default)/'full'
        * n_roots: Number of states printed in a single diagonalization process
    - cis-soc: Performing CIS calculation with spin-orbital coupling (SOC) included. See [hf-cis-soc.json](example/hf-cis-soc.json) as example. Currently only available for closed shell.
    - cis-soc_kwargs: Keyword argument passed to CIS with SOC.
        * n_roots: Number of states printed
- scan (optional): accept two types of input:
    - "variable":\[start:stop:step]: will create an array from start to (included) stop; Currently only accepts a single variable.
    - "i,j":math expression of variable: will change the value of element i,j (start from 1) of coordinates according to expression.
- verbose: full/normal/minimal/silent;

## Install

Python3 and numpy, scipy is required. Python module _mayavi_ is required only for plotting.

![](ch4.jpg)