from numba import cuda


@cuda.jit
def B3_S23(buffer, convoluted, somK):
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
def smoothLife(buffer, convoluted, somK):
    i, j = cuda.grid(2)
    if i < buffer.shape[0] and j < buffer.shape[1]:
        x = convoluted[i, j] / somK
        if x < 2./somK:
            buffer[i, j] = -1.
        elif x >= 2./8 and x < 3./somK:
            buffer[i, j] = 0.
        elif x >= 3./8 and x < 5./somK:
            buffer[i, j] = 1.
        else:
            buffer[i, j] = -1.


GROWTHS = {'B3_S23': B3_S23, 'smooth': smoothLife}
