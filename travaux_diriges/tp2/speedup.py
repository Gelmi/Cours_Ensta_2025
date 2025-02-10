import matplotlib.pyplot as plt
import numpy as np

processos = [1, 2, 3, 4]
tempo_serial = 1.3805
tempo_caso_1 = [1.4256, 0.7357, 0.5406, 0.5078]
tempo_caso_2 = [1.4383, 0.8340, 0.7432, 0.7868]
tempo_caso_3 = [None, 1.3077, 0.7168, 0.5341] 

def calcular_speedup(tempo_serial, tempos_paralelos):
    return [tempo_serial / t if t is not None else None for t in tempos_paralelos]

speedup_caso_1 = calcular_speedup(tempo_serial, tempo_caso_1)
speedup_caso_2 = calcular_speedup(tempo_serial, tempo_caso_2)
speedup_caso_3 = calcular_speedup(tempo_serial, tempo_caso_3)

plt.figure(figsize=(10, 6))

plt.plot(processos, speedup_caso_1, marker='o', label='Case 1', linestyle='-')

plt.plot(processos, speedup_caso_2, marker='s', label='Case 2', linestyle='--')

plt.plot(processos[1:], speedup_caso_3[1:], marker='^', label='Case 3', linestyle='-.')

plt.plot(processos, processos, marker='', label='Ideal Speedup', linestyle=':', color='gray')

plt.xlabel('Number of threads')
plt.ylabel('Speedup')
plt.title('Speedup for each case')
plt.legend()
plt.grid(True)
plt.xticks(processos)
plt.savefig("speedup_mandelbrot.png")
