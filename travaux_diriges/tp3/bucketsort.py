from mpi4py import MPI
import random
import numpy as np
from time import time

comm = MPI.COMM_WORLD.Dup()
nbp = comm.size
rank = comm.rank


def main():
    if rank == 0:
        random.seed()
        mean_val_bucket = 160000
        array = [random.randrange(0, 10000, 1)/10000 for _ in range(mean_val_bucket)]
        partitions = []
        n_elements = [0 for _ in range(nbp)]
        buckets = [[] for _ in range(nbp)]
        size = 1/nbp
        start = 0
        for i in range(nbp):
            partitions.append((start, start+size))
            start += size
        for element in array:
            for process in range(nbp):
                if element >= partitions[process][0] and element <= partitions[process][1]:
                    n_elements[process] += 1
                    buckets[process].append(element)
                    break
    else:
        n_elements = None
        buckets = None
    n_elements = comm.bcast(n_elements, root=0)
    partial_array = np.zeros(n_elements[rank])
    if rank == 0:
        buckets = np.hstack(buckets)
        # print(buckets)
    comm.Scatterv((buckets, n_elements), partial_array, root=0)
    deb = time()
    partial_array.sort()
    fin = time()
    print(f"Temps de tri par le thread {rank} : {fin-deb}")
    # print(partial_array)
    comm.Gatherv(partial_array, (buckets, n_elements), root=0)
    if rank == 0:
        print(buckets)


if __name__ == "__main__":
    main()
