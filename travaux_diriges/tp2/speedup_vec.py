import matplotlib.pyplot as plt
import numpy as np

processos = [1, 2, 3, 4]
tempo_serial = 0.0008299350738525391
tempo_caso_1 = [0.7459132671356201, 0.13793420791625977, 0.0641946792602539, 0.0366361141204834]
tempo_caso_2 = [0.0012798309326171875, 0.0006449222564697266, 0.0004849433898925781, 0.0003879070281982422]

def calcular_speedup(tempo_serial, tempos_paralelos):
    return [tempo_serial / t if t is not None else None for t in tempos_paralelos]

speedup_caso_1 = calcular_speedup(tempo_serial, tempo_caso_1)
speedup_caso_2 = calcular_speedup(tempo_serial, tempo_caso_2)

plt.figure(figsize=(10, 6))

plt.plot(processos, speedup_caso_1, marker='o', label='Case Columns', linestyle='-')

plt.plot(processos, speedup_caso_2, marker='s', label='Case Lines', linestyle='--')

plt.plot(processos, processos, marker='', label='Ideal Speedup', linestyle=':', color='gray')

plt.xlabel('Number of threads')
plt.ylabel('Speedup')
plt.title('Speedup for each case')
plt.legend()
plt.grid(True)
plt.xticks(processos)
plt.savefig("speedup_matvec.png")
