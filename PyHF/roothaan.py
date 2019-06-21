
import numpy as np 
import numpy.linalg as LA

def diagonal(F:np.array, S:np.array):
    """ Solve eigenvalue equation FC = eSC, where F and S are Hermitian matrices;
    Returns: e and C, sorted by increasing order
    """

    s, U = LA.eigh(S)
    X = U.dot(np.diag(s**-0.5))

    e, C2 = LA.eigh(X.T.dot(F.dot(X)))

    C = X.dot(C2)

    # sort
    idx = e.argsort()
    return e[idx], C[:,idx]


def build_fock(C, h, v, n_orbital):
    """ Generate Fock matrix for restricted HF
        F_ij = h_ij + \sum_{kl}{D_lk*(v_ijkl - 0.5*v_ilkj)}
    """
    D = 2*build_density_mat(C, n_orbital)

    F = np.zeros_like(h)

    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            F[i, j] = np.sum(D*(v[i,j,:,:].T - 0.5*v[i,:,:,j]))

    return F + h


def build_fock_u(C, h, v, n_orbital):
    """ Generate Fock matrix for unrestricted HF
        Fa_ij = h_ij + \sum_{kl}{D_lk*v_ijkl - Da_lk*v_ilkj}
        Fb_ij = h_ij + \sum_{kl}{D_lk*v_ijkl - Db_lk*v_ilkj}

        Both C and n_orbital require tuple (alpha,beta) here.
        Returns a tuple (Fa,Fb)
    """
    D_a = build_density_mat(C[0], n_orbital[0])
    D_b = build_density_mat(C[1], n_orbital[1])

    D = D_a + D_b

    F_a = np.zeros_like(h)
    F_b = np.zeros_like(h)

    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            F_a[i, j] = np.sum(D*v[i,j,:,:].T - D_a*v[i,:,:,j])
            F_b[i, j] = np.sum(D*v[i,j,:,:].T - D_b*v[i,:,:,j])

    return F_a + h, F_b + h


def fock2ro(F_a, F_b, n_a, n_b, Ac, Av, Ao, Bo):
    
    F = np.zeros_like(F_a)

    F[:n_b, n_b:n_a] = F_b[:n_b, n_b:n_a]
    F[n_b:n_a, n_a:] = F_a[n_b:n_a, n_a:]
    F[:n_b, n_a:] = 0.5*(F_a[:n_b, n_a:] + F_b[:n_b, n_a:])
    F = F + F.T

    F[:n_b, :n_b] = Ac * F_a[:n_b,:n_b] + (1-Ac) * F_b[:n_b,:n_b]
    F[n_a:, n_a:] = Av * F_a[n_a:, n_a:] + (1-Av) * F_b[n_a:,n_a:]
    F[n_b:n_a, n_b:n_a] = Ao * F_a[n_b:n_a, n_b:n_a] + Bo * F_b[n_b:n_a, n_b:n_a]

    return F
    

def build_density_mat(C, n_orbital):
    """ Generate density matrix D (not P) from state matrix.
    The eigenvalues of C's columns are assumed from smallest to largest.

    Args:
        C: coefficient matrix;
        n_orbital: number of occupied orbitals;
    """

    D = np.zeros_like(C)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            D[i, j] = sum([C[i, n]*C[j, n] for n in range(n_orbital)])

    return D
