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
    escape_radius: float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        z: complex
        iter: int

        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations

        z = 0
        for iter in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z))) / log(2)
                return iter
        return self.max_iterations

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3. / width
scaleY = 2.25 / height

def estimate_line_complexity(y):
    complexity = 0
    for x in range(0, width, 10):         
        c = complex(-2 + scaleX * x, -1.125 + scaleY * y)
        complexity += mandelbrot_set.count_iterations(c)
    return complexity

if rank == 0:
    line_complexities = [estimate_line_complexity(y) for y in range(height)]
    total_complexity = sum(line_complexities)
    target_complexity_per_process = total_complexity / nbp
    partitions = []
    current_complexity = 0
    start = 0
    for y in range(height):
        current_complexity += line_complexities[y]
        if current_complexity >= target_complexity_per_process:
            partitions.append((start, y + 1))
            start = y + 1
            current_complexity = 0
    if start < height:
        partitions.append((start, height))
    print(partitions)
else:
    partitions = None

partitions = globCom.bcast(partitions, root=0)

partial_convergence = np.zeros((partitions[rank][1] - partitions[rank][0], width), dtype=np.double)

deb = time()
for y in range(*partitions[rank]):
    for x in range(width):
        c = complex(-2 + scaleX * x, -1.125 + scaleY * y)
        partial_convergence[y - partitions[rank][0], x] = mandelbrot_set.convergence(c, smooth=True)
fin = time()
print(f"Temps du calcul de l'ensemble de Mandelbrot par le thread {rank} : {fin - deb}")

# Rassembler les résultats sur le processus 0
if rank == 0:
    convergence = np.zeros((height, width), dtype=np.double)
else:
    convergence = None

send_counts = [partitions[i][1] - partitions[i][0] for i in range(nbp)]
send_counts = np.array(send_counts) * width

globCom.Gatherv(partial_convergence, (convergence, send_counts), root=0)

# Constitution de l'image résultante :
if rank == 0:
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin - deb}")
    image.show()
