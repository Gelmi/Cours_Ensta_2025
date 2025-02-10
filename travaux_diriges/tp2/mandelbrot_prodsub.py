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

def compute_line(y):
    line_result = np.zeros(width, dtype=np.double)
    for x in range(width):
        c = complex(-2 + scaleX * x, -1.125 + scaleY * y)
        line_result[x] = mandelbrot_set.convergence(c, smooth=True)
    return line_result

if rank == 0:
    convergence = np.zeros((height, width), dtype=np.double)      
    lines_to_process = list(range(height))  
    current_line = 0

    deb = time()

    for worker in range(1, nbp):
        if current_line < height:
            globCom.send(current_line, dest=worker, tag=1)
            current_line += 1
        else:
            globCom.send(None, dest=worker, tag=1)   

    while current_line < height:
        status = MPI.Status()
        line_result = globCom.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        worker = status.Get_source()
        y, result = line_result
        convergence[y, :] = result

        if current_line < height:
            globCom.send(current_line, dest=worker, tag=1)
            current_line += 1
        else:
            globCom.send(None, dest=worker, tag=1)  

    for worker in range(1, nbp):
        status = MPI.Status()
        line_result = globCom.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        if line_result is not None:
            y, result = line_result
            convergence[y, :] = result

    fin = time()
    print(f"Temps total du calcul de l'ensemble de Mandelbrot : {fin - deb}")

    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin - deb}")
    image.show()

else:
    while True:
        y = globCom.recv(source=0, tag=1)
        if y is None:
            print("acabei")
            break  

        result = compute_line(y)

        globCom.send((y, result), dest=0, tag=2)
