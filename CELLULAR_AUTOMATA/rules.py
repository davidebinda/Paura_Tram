import math
from numba import cuda
import cupy as cp


@cuda.jit  # THE ORIGINAL THE LEGEND
def B3_S23(buffer, convoluted, somK, args):
    i, j = cuda.grid(2)
    if i < buffer.shape[0] and j < buffer.shape[1]:
        x = convoluted[i, j] / somK  # normalises convolution between 0 and 1
        if x < 2./somK:
            buffer[i, j] = -1.
        elif x >= 2./8 and x < 3./somK:
            buffer[i, j] = 0.
        elif x >= 3./8 and x < 4./somK:
            buffer[i, j] = 1.
        else:
            buffer[i, j] = -1.


@cuda.jit
def nextKernel(buffer, convoluted, somK, args):
    i, j = cuda.grid(2)
    if i < buffer.shape[0] and j < buffer.shape[1]:
        buffer[i, j] = convoluted[i, j] / somK


@cuda.jit
def lenia(buffer, convoluted, somK, args):
    i, j = cuda.grid(2)
    if i < buffer.shape[0] and j < buffer.shape[1]:
        normSom = convoluted[i, j] / somK

        mu = args[0]
        sigma = args[1]
        l = abs(normSom - mu)
        k = 2 * sigma**2
        buffer[i, j] = 2 * math.exp((-l**2) / k) - 1


GROWTHS = {'B3_S23': B3_S23, 'nextKernel': nextKernel, 'lenia': lenia}
