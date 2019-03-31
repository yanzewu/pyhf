
import json
import enum
import numpy as np

from .preproc import _getcwd

_norm2vec = lambda v:np.linalg.norm(v,axis=1)**2


class Basis:
    """ GTO Basis
    s-type:
        psi = d1*exp(-a1*(r-r0)^2) + d2*exp(-a2*(r-r0)^2) + ...
    p-type:
        psi = d1*(x-x0)*exp(-a1*(r-r0)^2) + d2*(x-x0)*exp(-a2*(r-r0)^2) + ...
    """

    TYPE_S = 0
    TYPE_P = 1

    OR_X = 0
    OR_Y = 1
    OR_Z = 2

    def __init__(self, type_, dataarr, origin=np.zeros(3), orientation=0, scale=1.0):
        """ Initialize a new Basis.
        Args:
            type_: Basis.TYPE_S, Basis.TYPE_P;
            dataarr: Nx2 array [[a1,d1],[a2,d2],...]
            origin: 1x3 array, location;
            orientation: Only for p orbital, Basis.OR_X, Basis.OR_Y, Basis.OR_Z;
            scale: a=>a*scale^2;
        """
        assert dataarr.shape[1] == 2, 'Invalid data type'

        self.type_ = type_
        self.data = dataarr
        self.origin = origin
        self.orientation = orientation

        for d in self.data:
            d[0]*=scale**2

        self.normalize()

    def normalize(self):
        """ Updating d by normalizing each Gaussian
        """
        if self.type_ == Basis.TYPE_S:
            for d in self.data:
                d[1] *= (2*d[0]/np.pi)**0.75
        elif self.type_ == Basis.TYPE_P:
            for d in self.data:
                d[1] *= (128*d[0]**5/np.pi**3)**0.25

    def evaluate_1d(self, x):
        """ Evaluate the basis over one axis;
        Args:
            x: array-like object
        Returns:
            1d array;
        """
        if self.type_ == Basis.TYPE_S:
            return np.sum([self.evaluate_1d_s_gauss(x,a,d) for a,d in self.data], axis=0)
        elif self.type_ == Basis.TYPE_P:
            return np.sum([self.evaluate_1d_p_gauss(x,a,d) for a,d in self.data], axis=0)
        
    def evaluate_3d(self, r):
        """ Evaluate the basis over the space;
        Args:
            r: nx3 array-like object;
        Returns:
            1d array;
        """
        if self.type_ == Basis.TYPE_S:
            return np.sum([self.evaluate_3d_s_gauss(r,a,d) for a,d in self.data], axis=0)
        elif self.type_ == Basis.TYPE_P:
            return np.sum([self.evaluate_3d_p_gauss(r,a,d) for a,d in self.data], axis=0)  

    def evaluate_1d_s_gauss(self, x, a, d):
        return d*np.exp(-a*(x-self.origin[0])**2);
        
    def evaluate_1d_p_gauss(self, x, a, d):
        return d*x*np.exp(-a*(x-self.origin[0])**2);

    def evaluate_3d_s_gauss(self, x, a, d):
        return d*np.exp(-a*_norm2vec(x-self.origin));
        
    def evaluate_3d_p_gauss(self, x, a, d):
        return d*(x[:,self.orientation].T-self.origin[self.orientation])*np.exp(-a*_norm2vec(x-self.origin));

    def __str__(self):
        if self.type_ == Basis.TYPE_S:
            return '+'.join(
                ['%g*exp(-%g*(r-(%g,%g,%g))^2)'%(
                    d,a,self.origin[0],self.origin[1],self.origin[2]
                ) for a,d in self.data])
        elif self.type_ == Basis.TYPE_P:
            str_ort = ['x','y','z'][self.orientation]

            return '+'.join(
                ['%g*(%s-%g)*exp(-%g*(r-(%g,%g,%g))^2)'%(
                    d,str_ort,self.origin[self.orientation],a,self.origin[0],self.origin[1],self.origin[2]
                ) for a,d in self.data])

    def __hash__(self):
        return hash((
            self.type_, 
            self.orientation, 
            self.origin[0],self.origin[1],self.origin[2],
            np.prod(self.data[:,0]), np.prod(self.data[:,1])))

    def __eq__(self, other):
        return self.type_ == other.type_ and self.orientation == other.orientation \
            and np.array_equal(self.origin,other.origin) and np.array_equal(self.data,other.data)


def construct_basis(basis_set, atom_charges, atom_coords):
    """ Construct a list of basis for certain atom configuration.
    Args:
        basis_set: name of basis;
        atom_charges: list of charge;
        atom_coords: list (or array) of 1x3 array
    Returns:
        list of basis.
    """

    charge2name = ['H','He','Li','Be','B','C','N','O','F']
    atom_names = [charge2name[c-1] for c in atom_charges]

    

    with open('%s/basis-%s.json' % (_getcwd(),basis_set), 'r') as fp:
        basis_set_data = json.load(fp)
    
    bases = []
    for atom, coord in zip(atom_names, atom_coords):
        for name, data in basis_set_data[atom]['basis'].items():
            if name[1] == 's':
                bases.append(Basis(Basis.TYPE_S, np.array(data), coord))
            elif name[1] == 'p':
                bases.append(Basis(Basis.TYPE_P, np.array(data), coord, Basis.OR_X))
                bases.append(Basis(Basis.TYPE_P, np.array(data), coord, Basis.OR_Y))
                bases.append(Basis(Basis.TYPE_P, np.array(data), coord, Basis.OR_Z))

    return bases
    