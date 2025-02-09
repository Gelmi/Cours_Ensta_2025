# Calcul de l'ensemble de Mandelbrot en python
import math
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank = globCom.rank


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

partition_size = math.floor(height/nbp)
temp_height = height
partition_start = 0
partitions = []
for cpuid in range(nbp-1):
    partitions.append((partition_start, partition_start+partition_size))
    temp_height -= partition_size
    partition_start += partition_size
partitions.append((partition_start, partition_start+temp_height))

partial_convergence = np.zeros((partitions[rank][1]-partitions[rank][0], width), dtype=np.double)

# Calcul de l'ensemble de mandelbrot :
deb = time()
for y in range(*partitions[rank]):
    for x in range(width):
        c = complex(-2 + scaleX*x, -1.125 + scaleY * y)
        partial_convergence[y - partitions[rank][0], x] = mandelbrot_set.convergence(c, smooth=True)

fin = time()
print(f"Temps du calcul de l'ensemble de Mandelbrot par le thread {rank} : {fin-deb}")

if rank == 0:
    convergence = np.zeros((height, width), dtype=np.double)
else:
    convergence = None

globCom.Gather(partial_convergence, convergence, root=0)

# Constitution de l'image résultante :
if rank == 0:
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()

