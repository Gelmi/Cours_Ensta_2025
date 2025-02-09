# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
import math
from time import time

globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank = globCom.rank

# Dimension du problème (peut-être changé)
dim = 120

# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
# print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
# print(f"u = {u}")
# print(u.shape)


def calculparCol(A, u, nbp, rank):
    # N loc
    Nloc = math.floor(dim/nbp)

    # Get columns
    columns = A[:, rank*Nloc:(rank+1)*Nloc]

    partial_mult = np.zeros((Nloc, dim))

    deb = time()
    for i, line in enumerate(columns):
        for j, element in enumerate(line):
            partial_mult[j][i] = line[j] * u[Nloc*rank + j]
    fin = time()
    print(f"Temps du calcul par le thread {rank} : {fin-deb}")
    if rank == 0:
        mult = np.zeros((dim, dim))
    else:
        mult = None

    globCom.Gather(partial_mult, mult, root=0)

    if rank == 0:
        deb2 = time()
        v = np.zeros(dim)
        for i, line in enumerate(mult.T):
            v[i] = np.sum(line)
        fin2 = time()
        print(f"Temps de constitution de le vecteur : {fin2-deb2}")
        print(f"Temps de calcul total : {fin2-deb}")


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
    print(f"Temps du calcul par ligne : {fin-deb}")
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
        print(f"Temps de calcul NP: {fin-deb}")


calcul(A, u, rank)
calculParLin(A, u, nbp, rank)
