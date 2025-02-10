# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
import math
from time import time

globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank = globCom.rank

# Dimension du problème (peut-être changé)
dim = 1200

# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
# print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
# print(f"u = {u}")
# print(u.shape)


def calculParCol(A, u, nbp, rank):
    # N loc
    Nloc = math.floor(dim/nbp)

    # Get columns
    columns = A[:, rank*Nloc:(rank+1)*Nloc]

    u_local = u[rank*Nloc:(rank+1)*Nloc]

    deb = time()
    partial_mult = np.dot(np.diag(u_local), columns.T) 
    fin = time()
    print(f"(Col) Temps du calcul par le thread {rank} : {fin-deb}")
    if rank == 0:
        mult = np.zeros((dim, dim))
    else:
        mult = None

    globCom.Gather(partial_mult, mult, root=0)

    if rank == 0:
        v = np.zeros(dim)
        for i, line in enumerate(mult.T):
            v[i] = np.sum(line)

def calculParLin(A, u, nbp, rank):
    # N loc
    Nloc = math.floor(dim/nbp)

    # Get columns
    rows = A[rank*Nloc:(rank+1)*Nloc, :]

    partial_mult = np.zeros(Nloc)
    deb = time()
    for i in range(Nloc):
        partial_mult[i] = rows[i].dot(u)
    fin = time()
    print(f"(Lin) Temps du calcul par ligne : {fin-deb}")
    if rank == 0:
        mult = np.zeros(dim)
    else:
        mult = None

    globCom.Gather(partial_mult, mult, root=0)


def calcul(A, u, rank):
    if rank == 0:
        deb = time()
        v = A.dot(u)
        fin = time()
        print(f"(NP) sTemps de calcul: {fin-deb}")


# calcul(A, u, rank)
calculParCol(A, u, nbp, rank)
calculParLin(A, u, nbp, rank)
