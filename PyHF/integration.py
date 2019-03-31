""" Calcualte the integration of basis
"""

import numpy as np
from scipy.special import hyp1f1

from . import basis

_norm2 = lambda v:np.inner(v,v)

# TODO: cutoff 

def _standardize_orbital_input(r1, r2, or1, or2, p1:int, p2:int):
    p1arr = np.zeros(3, dtype=int)
    p2arr = np.zeros(3, dtype=int)

    p1arr[or1] = p1
    p2arr[or2] = p2

    r12 = r2 - r1

    return p1arr, p2arr, r12


def evaluate_overlap(lhs:basis.Basis, rhs:basis.Basis):
    """ S_uv = <u|v>;
    """
    
    orbinput = _standardize_orbital_input(
                lhs.origin, rhs.origin,
                lhs.orientation, rhs.orientation,
                lhs.type_, rhs.type_)

    s = 0
    for a1,d1 in lhs.data:
        for a2,d2 in rhs.data:
            s += d1*d2*_pp_3d(a1, a2, *orbinput)

    return s


def evaluate_kernel(lhs:basis.Basis, rhs:basis.Basis, nuclear_coords, nuclear_charges):
    """ h_uv = <u|h|v>
    """

    orbinput = _standardize_orbital_input(
                lhs.origin, rhs.origin,
                lhs.orientation, rhs.orientation,
                lhs.type_, rhs.type_)

    s = 0
    for a1,d1 in lhs.data:
        for a2,d2 in rhs.data:
            s += -0.5*d1*d2*_pd2p_3d(a1, a2, *orbinput)
            for rc, z in zip(nuclear_coords, nuclear_charges):
                s -= d1*d2*z*_nu_coul_3d(a1, a2, rc-lhs.origin, *orbinput)

    return s


_ele_coul_global_cache = {}
def evaluate_2electron_coulomb(lhs1:basis.Basis, lhs2:basis.Basis, rhs1:basis.Basis, rhs2:basis.Basis):
    """ v_l1l2r1r2 = (lhs1 lhs2 | rhs1 rhs2)
    """
        
    _idx = (lhs1, lhs2, rhs1, rhs2)
    if _idx in _ele_coul_global_cache:
        return _ele_coul_global_cache[_idx]

    orbinput1 = _standardize_orbital_input(
                lhs1.origin, lhs2.origin,
                lhs1.orientation, lhs2.orientation,
                lhs1.type_, lhs2.type_)

    orbinput2 = _standardize_orbital_input(
            rhs1.origin, rhs2.origin,
            rhs1.orientation, rhs2.orientation,
            rhs1.type_, rhs2.type_)

    s = 0
    for a1,d1 in lhs1.data:
        for a2,d2 in lhs2.data:
            for a3,d3 in rhs1.data:
                for a4,d4 in rhs2.data:
                    s += d1*d2*d3*d4*_ele_coul_3d(a1,a2,a3,a4,*orbinput1,*orbinput2, rhs1.origin-lhs1.origin)

    _ele_coul_global_cache[_idx] = s
    return s

def _pp_3d(a1, a2, p1, p2, r12):

    return np.product([
        _pp_1d(a1, a2, r12[k], p1[k], p2[k]) for k in range(3)
    ])
    
def _pd2p_3d(a1, a2, p1, p2, r12):

    return np.sum([
        _pd2p_1d(a1,a2,r12[k],p1[k],p2[k])*_pp_1d(a1,a2,r12[k-1],p1[k-1],p2[k-1])*_pp_1d(a1,a2,r12[k-2],p1[k-2],p2[k-2]) for k in range(3)
    ])

def _pp_1d(a1, a2, x12, p1, p2):
    """ Evaluate the integral
        (x-x12)^p1*x^p2*exp(-a1*(x-x12)^2-a2*x^2)
    """
    
    return np.sqrt(np.pi/(a1+a2))*Hermite(p1,p2,0,x12,a1,a2)


def _pd2p_1d(a1, a2, x12, p1, p2):
    """ Evaluate the integral
        (x-x12)^p1*exp(-a1*(x-x12)^2)*d^2(x^p2*exp(-a2*x^2))/dx^2
    """

    return p2*(p2-1)*_pp_1d(a1,a2,x12,p1,p2-2) - \
        2*a2*(2*p2+1)*_pp_1d(a1,a2,x12,p1,p2) + \
        4*a2**2*_pp_1d(a1,a2,x12,p1,p2+2)


def _nu_coul_3d(a1, a2, rc, p1, p2, r12):
    """ Evaluate the nuclear colomb integral
        1/(r-rc)*G(r,a1,p1)*G(r-r12,a2,p2)
    """

    rpc = (a2*r12) / (a1 + a2) - rc
    rpc2 = _norm2(rpc)

    s = 0.0
    for t,u,v in np.ndindex(int(p1[0]+p2[0]+1),int(p1[1]+p2[1]+1),int(p1[2]+p2[2]+1)):
        s += Hermite(p1[0],p2[0],t,r12[0],a1,a2) * \
        Hermite(p1[1],p2[1],u,r12[1],a1,a2) * \
        Hermite(p1[2],p2[2],v,r12[2],a1,a2) * \
        _Hermite_coulomb(t,u,v,0,a1+a2,rpc, rpc2)
    return 2*np.pi/(a1+a2)*s


def _ele_coul_3d(a1, a2, a3, a4, p1, p2, r12, p3, p4, r34, r13):
    """ Evaluate the integral (over ra, rb)
        1/(ra-rb)*G(ra,a1,p1)*G(ra-r12,a2,p2)*G(rb-r13,a3,p3)*G(rb-r13-r34,a4,p4)
    """

    p = (a1+a2)*(a3+a4)/(a1+a2+a3+a4)

    rpq = (a2*r12)/(a1+a2) - (a4*r34)/(a3+a4) - r13
    rpq2 = _norm2(rpq)

    s = 0.0
    # TODO: Vectorization; Multiprocess
    
    for t1,u1,v1,t2,u2,v2 in np.ndindex(int(p1[0]+p2[0]+1),int(p1[1]+p2[1]+1),int(p1[2]+p2[2]+1),
        int(p3[0]+p4[0]+1),int(p3[1]+p4[1]+1),int(p3[2]+p4[2]+1)):
        s += \
            (-1)**(t2+u2+v2) *\
            Hermite(p1[0],p2[0],t1,r12[0],a1,a2) * \
            Hermite(p1[1],p2[1],u1,r12[1],a1,a2) * \
            Hermite(p1[2],p2[2],v1,r12[2],a1,a2) * \
            Hermite(p3[0],p4[0],t2,r34[0],a3,a4) * \
            Hermite(p3[1],p4[1],u2,r34[1],a3,a4) * \
            Hermite(p3[2],p4[2],v2,r34[2],a3,a4) * \
            _Hermite_coulomb(t1+t2,u1+u2,v1+v2,0,p,rpq,rpq2)

    return 2*np.pi**2.5/((a1+a2)*(a3+a4)*np.sqrt(a1+a2+a3+a4))*s

def _Boys(n, T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)


_Hermite_cache = {}
def Hermite(p1, p2, t, x12, a1, a2):
    """ Return Hermite coeff of (x-A)^p1*(x-B)^p2*exp(-a1*(x-A)^2-a2*(x-B)^2)
        at
        Hermite H_t(x-xp)*exp(-(a+b)*(x-xp)^2)

        Here x12 = x2 - x1;
    """


    if t < 0 or t > p1 + p2:
        return 0.0

    _index = (p1,p2,t,x12,a1,a2)

    if _index in _Hermite_cache:
        return _Hermite_cache[_index]

    A = a1 + a2
    B = (a1*a2)/(a1 + a2)

    if p1 == p2 == t == 0:
        ret = np.exp(-B*x12**2)

    elif p2 == 0:   # i,j => i-1,j
        ret = Hermite(p1-1,p2,t-1,x12,a1,a2)/(2*A) + \
            Hermite(p1-1,p2,t,x12,a1,a2)*B*x12/a1 + \
            Hermite(p1-1,p2,t+1,x12,a1,a2)*(t+1)
    
    else:
        ret = Hermite(p1,p2-1,t-1,x12,a1,a2)/(2*A) - \
            Hermite(p1,p2-1,t,x12,a1,a2)*B*x12/a2 + \
            Hermite(p1,p2-1,t+1,x12,a1,a2)*(t+1)    

    _Hermite_cache[_index] = ret
    return ret


_Hermite_coulomb_cache = {}
def _Hermite_coulomb(t, u, v, n, p, rpc, rpc2):

    _index = (t,u,v,n,p,rpc[0],rpc[1],rpc[2])
    if _index in _Hermite_coulomb_cache:
        return _Hermite_coulomb_cache[_index]

    if t == u == v == 0:
        ret = np.power(-2*p, n)*_Boys(n, p * rpc2)
    elif t == u == 0:
        ret = rpc[2]*_Hermite_coulomb(t,u,v-1,n+1,p,rpc,rpc2)
        if v > 1:
            ret += (v-1)*_Hermite_coulomb(t,u,v-2,n+1,p,rpc,rpc2)
    elif t == 0:
        ret = rpc[1]*_Hermite_coulomb(t,u-1,v,n+1,p,rpc,rpc2)
        if u > 1:
            ret += (u-1)*_Hermite_coulomb(t,u-2,v,n+1,p,rpc,rpc2)
    else:
        ret = rpc[0]*_Hermite_coulomb(t-1,u,v,n+1,p,rpc,rpc2)
        if t > 1:
            ret += (t-1)*_Hermite_coulomb(t-2,u,v,n+1,p,rpc,rpc2)

    _Hermite_coulomb_cache[_index] = ret
    return ret

