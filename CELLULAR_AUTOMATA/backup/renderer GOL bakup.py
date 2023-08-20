import pygame
import time
import numpy as np
from numba import cuda

WIDTH = 800
HEIGHT = 800
SCREEN_REALESTATE = (WIDTH, HEIGHT)
RES = 10
COLS = int(WIDTH/RES)
ROWS = int(HEIGHT/RES)

#####################################################################
# PYGAME INITIALIZE #################################################


pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode(SCREEN_REALESTATE)
font = pygame.font.SysFont("Arial", 18, bold=True)
FRAMES = 120


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


def nextState(prevMatx):
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

#####################################################################


matrix = np.round(np.random.random((COLS, ROWS, 3)))
bufferMatx = np.copy(matrix)
matrix[:, :, 1] = 0.
matrix[:, :, 2] = 1.

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    render(matrix)

    nextState(matrix)

    # running = False
    # time.sleep(5)

    ################################################################
