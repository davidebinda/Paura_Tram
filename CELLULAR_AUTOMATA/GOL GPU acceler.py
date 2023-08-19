import pygame
import time
import numpy as np
from numba import cuda

WIDTH = 1300
HEIGHT = 800
SCREEN_REALESTATE = (WIDTH, HEIGHT)
RES = 5
COLS = int(WIDTH/RES)
ROWS = int(HEIGHT/RES)

print('[NUMBER OF CELLS] ', COLS*ROWS)

#####################################################################
# PYGAME INITIALIZE #################################################


pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode(SCREEN_REALESTATE)
font = pygame.font.SysFont("Arial", 18, bold=True)
FRAMES = 60


def fps_counter():
    fps = str(int(clock.get_fps()))
    pygame.display.set_caption(f'ROOTS || {fps}')

#####################################################################


def render(m):
    screenarray = m * 255.0
    surface = pygame.surfarray.make_surface(screenarray)
    surf_trans = pygame.transform.scale(surface, SCREEN_REALESTATE)
    screen.blit(surf_trans, (0, 0))

    fps_counter()

    pygame.display.flip()
    clock.tick(FRAMES)


def neigCount(m, x, y):
    sum = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            col = (x + i + COLS) % COLS
            row = (y + j + ROWS) % ROWS
            sum += m[col, row, 0]

    sum -= m[x, y, 0]

    return sum


def nextState_CPU(prevMatx):
    global matrix
    nextMatx = np.copy(prevMatx)

    # rules
    for i in range(prevMatx.shape[0]):
        for j in range(prevMatx.shape[1]):

            sum = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    col = (x + i + COLS) % COLS
                    row = (y + j + ROWS) % ROWS
                    sum += prevMatx[col, row, 0]

            sum -= prevMatx[i, j, 0]

            neighbours = sum

            if prevMatx[i, j, 0] == 1:
                if neighbours < 2 or neighbours > 3:
                    nextMatx[i, j, 0] = 0
            else:
                if neighbours == 3:
                    nextMatx[i, j, 0] = 1

    matrix = np.copy(nextMatx)


@cuda.jit
def nextState(m, buffer):

    i, j = cuda.grid(2)
    if i < buffer.shape[0] and j < buffer.shape[1]:

        neighbours = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                col = (i + x + COLS) % COLS
                row = (j + y + ROWS) % ROWS
                neighbours += m[col, row, 0]

        neighbours -= m[i, j, 0]

        if m[i, j, 0] == 1:
            if neighbours < 2 or neighbours > 3:
                buffer[i, j, 0] = 0
        else:
            if neighbours == 3:
                buffer[i, j, 0] = 1

        '''buffer[i, j, 0] = 0
        if i == 1:
            buffer[i, j, 0] = 1'''


#####################################################################


matrix = np.round(np.random.random((COLS, ROWS, 3)))
matrix[:, :, 1] = 0.
matrix[:, :, 2] = 1.

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    bufferMatx = np.copy(matrix)

    render(matrix)

    # CUDASHIT
    threadsperblock = (32, 32)
    blockspergrid_x = int(np.ceil(COLS / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(ROWS / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    nextState[blockspergrid, threadsperblock](matrix, bufferMatx)

    matrix = bufferMatx
    # nextState_CPU(matrix)
    # running = False
    # time.sleep(5)

    ################################################################
