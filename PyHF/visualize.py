
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

try:
    from mayavi import mlab
except ImportError:
    import sys
    print('Cannot import graphics module, plot function will not work.', file=sys.stderr)

from . import postproc


def _creating_mesh(origins, atom_ext, sample_step=0.1):
    mins = np.min(origins, axis=0)
    maxs = np.max(origins, axis=0)

    x = np.arange(mins[0]-atom_ext, maxs[0]+atom_ext, sample_step)
    y = np.arange(mins[1]-atom_ext, maxs[1]+atom_ext, sample_step)
    z = np.arange(mins[2]-atom_ext, maxs[2]+atom_ext, sample_step)
    return np.meshgrid(x,y,z, indexing='ij')


def _plot_mesh(X, Y, Z, rho, contour, color):

    src = mlab.pipeline.scalar_field(X, Y, Z, np.reshape(rho, X.shape))
    s = mlab.pipeline.iso_surface(src, contours=[contour], opacity=0.5, color=color)


def plot_orbital(c, bases, color=(0,0,1), atom_ext=2.0, sample_step=0.1, contour=0.01):

    X,Y,Z = _creating_mesh(np.array([b.origin for b in bases]), atom_ext, sample_step)
    
    R = np.vstack((np.ravel(X),np.ravel(Y),np.ravel(Z))).T

    psi = np.sum([c[i]*bases[i].evaluate_3d(R) for i in range(len(bases))], axis=0)
    rho = psi**2

    _plot_mesh(X,Y,Z,rho, contour=contour, color=color)
    

def plot_orbital_list(orbital_idx, C, bases, postfix='', **kwargs):
    """ Visualizing all orbitals
    """

    for j, i in enumerate(orbital_idx):
        if i-1 >= C.shape[1] or i-1 < 0:
            print('Orbital %d does not exist. Skipping' % i)
        else:
            plot_orbital(C[:,i-1], bases, color=cm.tab10(j)[:3], **kwargs)
            mlab.text(0.05, 0.1*j, str(i) + postfix, width=0.2, color=cm.tab10(j)[:3])

def plot_charge_density(D, bases, atom_ext=2.0, sample_step=0.1, contour=0.01):

    X,Y,Z = _creating_mesh(np.array([b.origin for b in bases]), atom_ext, sample_step)
    
    R = np.vstack((np.ravel(X),np.ravel(Y),np.ravel(Z))).T

    psi_bases = [b.evaluate_3d(R) for b in bases]

    rho = np.zeros(len(R))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            rho += D[i,j]*psi_bases[i]*psi_bases[j]

    src = mlab.pipeline.scalar_field(X, Y, Z, np.reshape(rho, X.shape))
    v = mlab.pipeline.volume(src, vmin=0, vmax=contour*10)
    mlab.colorbar(v, title='Charge density', orientation='vertical', nb_labels=6, label_fmt='%.2f', nb_colors=24)


def plot_backbone(atom_charges, atom_coords):

    charge2color = [
        (1.0,1.0,1.0),
        (0.9,0.9,0.9),
        (0.6,0.0,1.0),
        (0.0,0.4,0.0),
        (1.0,0.8,0.5),
        (0.1,0.1,0.1),
        (0.0,0.0,0.5),
        (1.0,0.2,0.0),
        (0.1,0.9,0.1)
    ]

    charge2radius = [
        0.2,0.3, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
    ]

    for c,r in zip(atom_charges, atom_coords):
        mlab.points3d(r[0],r[1],r[2], charge2radius[c-1], scale_factor=1, color=charge2color[c-1], resolution=10)


def ui_plot(hftype, C, bases, n_orbital, atom_charges, atom_coords, name=''):
    """ Create a prompt session for orbital plotting.
        hftype: 'rhf'/'uhf';
        C, bases, n_orbital, atom_charges, atom_coords: see rhf() or post_analysis_rhf();
        name: name of the graph;

        Available commands:
        - "charge"
        - number representing orbital: 1,2,3,... in RHF, 1a,2b,... in UHF; Can be seperated by space.
        - "homo": will show number of HOMO
        - "q": exit
    """

    print('Entering plotting session. Enter number to plot orbital. Enter q to exit.')

    _plotargs = {
        'atom_ext':2.0,
        'sample_step':0.1,
        'contour':0.01
    }

    while True:
        
        cmd = input('> ')

        if cmd == '':
            continue

        elif cmd == 'q' or cmd == 'exit':
            break
        
        elif cmd == 'homo':
            if hftype == 'rhf':
                print(n_orbital)
            elif hftype == 'uhf':
                print('%da, %db' % (n_orbital[0], n_orbital[1]))

        elif cmd[:3] == 'set':
            cmdsplit = cmd.split()
            try:
                _plotargs[cmdsplit[1]] = float(cmdsplit[2])
            except KeyError:
                print('Variable %s does not exist' % cmdsplit[1])
                continue
            except Exception as e:
                print(e)
                continue

        else:
            
            mlab.figure(name, bgcolor=(1,1,1))
            plot_backbone(atom_charges, atom_coords)

            if cmd == 'charge':
                D = postproc.density_matrix(C, n_orbital)
                if isinstance(D, tuple):
                    D = D[0] + D[1]
                plot_charge_density(D, bases, **_plotargs)

            else:

                try:

                    if hftype == 'rhf':
                        orbital_list = list(map(int, cmd.split()))

                    elif hftype == 'uhf':
                        cmdsplit = cmd.split()
                        orbital_list_a = list(map(int, (r[:-1] for r in cmdsplit if r[-1] == 'a')))
                        orbital_list_b = list(map(int, (r[:-1] for r in cmdsplit if r[-1] == 'b')))
                except Exception as e:
                    print(e)
                    continue

                if hftype == 'rhf':
                    plot_orbital_list(orbital_list, C, bases, **_plotargs)

                elif hftype == 'uhf':
                    plot_orbital_list(orbital_list_a, C[0], bases, postfix='a', **_plotargs)
                    plot_orbital_list(orbital_list_b, C[1], bases, postfix='b', **_plotargs)
                
            mlab.orientation_axes()
            mlab.show()
