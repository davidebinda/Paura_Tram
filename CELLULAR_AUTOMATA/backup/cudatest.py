from numba import cuda
import numpy as np
import cupy as cp
import time

print('pippo')


@cuda.jit
def matmul(C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        # tmp = 0.
        # for k in range(A.shape[1]):
        #    tmp += A[i,k] * B[k,j]
        C[i, j] += 1.0


def matadd(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i, j] += 1


cp.random.seed(69)
A = cp.random.uniform(1, 10, size=(1000, 1000), dtype=np.float64)
B = cp.random.uniform(1, 10, size=(1000, 1000), dtype=np.float64)
C = cp.zeros((1000, 1000), dtype=np.float64)

# block ha 16x16 threads, tipicamente 128-512 threads/block
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(C.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(C.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

print(blockspergrid)
print(f"max kernel number: {threadsperblock[0]*blockspergrid_x}")  # ? ##

# proviamo con la cpu
D = cp.zeros((10, 10), dtype=np.float64)

print("CALCULATING ON CPU...")
start = time.time()

# LETS RUN THIS BITCH
for i in range(10):
    matadd(D)

print(f"TIME ELAPSED: {time.time() - start}")

print(f"\n MATRICE: {D}")

#########################################################################
#########################################################################
#########################################################################

C = np.zeros((1000, 1000))

d_C = cuda.to_device(C)

# proviamo con la gpu
print("\nCALCULATING ON GPU...")
start = time.time()

# LETS RUN THIS BITCH
for i in range(10):
    matmul[blockspergrid, threadsperblock](d_C)

C = d_C.copy_to_host()

print(f"TIME ELAPSED: {time.time() - start}")

print(f"\n MATRICE: {C}")

###### STORE ARRAY IN GPU #######

'''array_cpu = np.random.randint(0, 10, size=(1000,1000)) # geerare array con nupy
d_array = cuda.to_device(array_cpu) # mandare array alla gpu

#increment_by_one(a)
print(d_array)
print(d_array.copy_to_host()) # copiare array da gpu a cpu'''
