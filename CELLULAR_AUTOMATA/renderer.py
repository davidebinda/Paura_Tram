import pygame
import cupyx.scipy.signal as sn
import numpy as np
import cupy as cp
from numba import cuda
from rules import GROWTHS
from matplotlib import cm
from tost import rasterize
import random

WIDTH = 1600
HEIGHT = 890
'''WIDTH = 1100
HEIGHT = 600'''
SCREEN_REALESTATE = (WIDTH, HEIGHT)
RES = 5
COLS = int(WIDTH/RES)
ROWS = int(HEIGHT/RES)
V_RES = RES
V_COLS = int(WIDTH/V_RES)
V_ROWS = int(HEIGHT/V_RES)
GROWTH_FUNC = 'B3_S23'
GROWTH_FUNC = 'lenia'

COLOR_PRECISION = 10000
COLORS = cm.jet(np.linspace(0, 1, COLOR_PRECISION))
print('[COLOR PALLETTE]\n', COLORS)

print('[NUMBER OF CELLS] ', COLS*ROWS)
print('[MATRIX DIMS] ', COLS, ' x ', ROWS)

#####################################################################
# PYGAME INITIALIZE #################################################


pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode(SCREEN_REALESTATE)
font = pygame.font.SysFont("JetBrains Mono", 18, bold=True)
FRAMES = 60


def fps_counter():
    fps = str(int(clock.get_fps()))
    pygame.display.set_caption(f'ROOTS || {fps}')

#####################################################################


# get color from jet colormap of cell value
@cuda.jit
def getColors(values, cols):
    i, j = cuda.grid(2)

    if i < values.shape[0] and j < values.shape[1]:
        index = int(values[i, j, 0] * COLOR_PRECISION)

        if index == COLOR_PRECISION:  # plz dont bully me for this
            index -= 1

        values[i, j, 0] = cols[index, 0] * 255
        values[i, j, 1] = cols[index, 1] * 255
        values[i, j, 2] = cols[index, 2] * 255


# well... it renders
def render(m):
    screenarray = cp.zeros((V_COLS, V_ROWS, 3))
    # m = m.get()

    y_min = int(np.floor((ROWS - V_ROWS) / 2))
    y_max = y_min + V_ROWS
    x_min = int(np.floor((COLS - V_COLS) / 2))
    x_max = x_min + V_COLS

    # copies matrix in screenarray R channel
    screenarray[:, :, 0] = m[x_min:x_max, y_min:y_max]

    '''print('ROWS ', ROWS, '\tCOLS ', COLS)
    print('V_ROWS ', V_ROWS, '\tV_COLS ', V_COLS)
    print('x: ', x_min, x_max)
    print('y: ', y_min, y_max)
    print('screen ', screenarray.shape[0],
          screenarray.shape[1], '\tmatrix ', m.shape[0], m.shape[1])'''

    # CUDASHIT
    threadsperblock = (32, 32)
    blockspergrid_x = int(cp.ceil(V_COLS / threadsperblock[0]))
    blockspergrid_y = int(cp.ceil(V_ROWS / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # print(screenarray[2, 1, 0], int(screenarray[2, 1, 0] * COLOR_PRECISION))
    # i have the power of god and gpu on my side
    getColors[blockspergrid, threadsperblock](
        screenarray, cp.array(COLORS))

    # renders array
    screenarray = screenarray.get()
    surface = pygame.surfarray.make_surface(screenarray)
    surf_trans = pygame.transform.scale(surface, SCREEN_REALESTATE)
    screen.blit(surf_trans, (0, 0))

    # renders name of pattern
    text_surface = font.render(file, True, (255, 0, 0))
    screen.blit(text_surface, (10, 10))

    fps_counter()

    pygame.display.flip()
    clock.tick(FRAMES)


# loads ssv | mode: k = kernel, m = matrix
def loadSSV(file, mode):
    # try:
    values = []

    with open(file, "r") as f:

        lines = f.readlines()

        # gets width and height
        size = lines[0].split(',')
        size = [x.strip() for x in size]

        width = int(size[0].split(' ')[-1])
        height = int(size[1].split(' ')[-1])

        lines = lines[1:]

        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].strip()
            line = lines[i].split(' ')
            line = [float(s) for s in line]
            values.append(line)

    if mode == 'm':
        # black matrix
        m = cp.zeros((COLS, ROWS))

        matxCenter = (
            # il meno 1 qui sotto serve perche sono un pirla e ho
            # fatto na cosa nel convertitore rle ssv che nn mi ricordo
            # come cambiare quindi ho messo questo qui nn se ne accorgerà mai nessuno
            int(COLS/2) - int(width/2) - 1,
            int(ROWS/2) - int(height/2))

        for i in range(len(values)):
            for j in range(len(values[i])):
                x = (matxCenter[1] + i + ROWS) % ROWS
                y = (matxCenter[0] + j + COLS) % COLS

                m[y, x] = values[i][j]

        return m

    elif mode == 'k':
        # black matrix
        m = cp.zeros((width, height))

        for i in range(height):
            for j in range(width):
                m[j, i] = values[i][j]

        return m, float(cp.sum(m))
    # except:
    # print('[NO SSV FILE -> RANDOM MODE]')


def sprinkleMatx(r):
    m = cp.zeros((COLS, ROWS))
    randArea = cp.random.random((np.clip(r+2, 1, COLS), np.clip(r+2, 1, ROWS)))

    print(f'[M] {m.shape[0]}x{m.shape[1]}')
    print(f'[randArea] {randArea.shape[0]}x{randArea.shape[1]}')

    mW = m.shape[0]
    mH = m.shape[1]
    areaW = randArea.shape[0]

    for i in range(20):
        xOffset = random.randint(-r*6, r*6)
        yOffset = random.randint(-r*6, r*6)

        left = (mW // 2) - (areaW // 2) + xOffset
        left = max(min(left, mW - areaW), 0)
        right = left + areaW
        up = (mH // 2) - (areaW // 2) + yOffset
        up = max(min(up, mH - areaW), 0)
        down = up + areaW

        print(f'  {up}')
        print(f'{left}  {right}')
        print(f'  {down}')

        m[left:right, up:down] = randArea

    return m

#####################################################################


# by default loads last SSV added to record
files = []
filesIndex = -1
with open('SSV/RLEs/record.txt', 'r') as f:
    files = f.readlines()

files = [x.strip('\n') for x in files]
file = files[filesIndex][:-4]

print('[SIMULATING LAST ADDED TO RECORD]->', file)

# loads default SSV file
matrix = loadSSV("SSV/" + file + ".ssv", mode='m')
# KERNEL BE LIKE (not used in lenia mode)
kernel, somK = loadSSV('SSV/KERNELs/GOL.kernel.ssv', mode='k')

##########################################
# we are using lenia so this is the correct kernel
mu = .15
sigma = .017
kernelRadius = 13
kernel, somK = rasterize(kernelRadius)

#########################################

# OMG THE LOOP WOW SO COOL
prevMat = cp.copy(matrix)
p = True
running = True
paused = True
forwOnes = False
while running:
    # gets inputs
    if True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # keyboard inputs
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # pauses simulation
                    paused = not paused

                elif event.key == pygame.K_RIGHT:
                    # goes one step forward
                    paused = False
                    forwOnes = True

                elif event.key == pygame.K_LEFT:
                    # goes as many steps back as in memory
                    matrix = cp.copy(prevMat)

                elif event.key == pygame.K_r:
                    matrix = sprinkleMatx(kernelRadius)
                    file = ''
                elif event.key == pygame.K_DOWN:
                    # loads next SSV from record
                    filesIndex = (filesIndex + 1) % len(files)
                    matrix = loadSSV(
                        "SSV/" + files[filesIndex][:-4] + ".ssv", mode='m')
                    file = files[filesIndex][:-4]
                    pastMemories = []
                    pastIndex = -1

                elif event.key == pygame.K_UP:
                    # loads next SSV from record
                    filesIndex = (filesIndex - 1) % len(files)
                    matrix = loadSSV(
                        "SSV/" + files[filesIndex][:-4] + ".ssv", mode='m')
                    file = files[filesIndex][:-4]
                    pastMemories = []
                    pastIndex = -1

                elif event.key == pygame.K_PLUS:
                    V_RES += 1
                    V_COLS = int(WIDTH/V_RES)
                    V_ROWS = int(HEIGHT/V_RES)
                elif event.key == pygame.K_MINUS:
                    V_RES -= 1
                    if V_RES < RES:
                        V_RES = RES
                    V_COLS = int(WIDTH/V_RES)
                    V_ROWS = int(HEIGHT/V_RES)

        # movement handler
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] == True:
            matrix = cp.roll(matrix, +2, axis=1)
        if keys[pygame.K_s] == True:
            matrix = cp.roll(matrix, -2, axis=1)
        if keys[pygame.K_a] == True:
            matrix = cp.roll(matrix, +2, axis=0)
        if keys[pygame.K_d] == True:
            matrix = cp.roll(matrix, -2, axis=0)
        # FAST ZOOM
        if keys[pygame.K_PLUS] == True and keys[pygame.K_LCTRL] == True:
            V_RES += 1
            V_COLS = int(WIDTH/V_RES)
            V_ROWS = int(HEIGHT/V_RES)
        if keys[pygame.K_MINUS] == True and keys[pygame.K_LCTRL] == True:
            V_RES -= 1
            if V_RES < RES:
                V_RES = RES
            V_COLS = int(WIDTH/V_RES)
            V_ROWS = int(HEIGHT/V_RES)

    if p:
        print('[MATRIX]\n', matrix)

    # copia mtx in buffer e fa convoluzione
    bufferMatx = cp.copy(matrix)

    convMatx = sn.convolve2d(matrix, kernel, mode='same', boundary='wrap')

    # CUDASHIT
    threadsperblock = (32, 32)
    blockspergrid_x = int(cp.ceil(COLS / threadsperblock[0]))
    blockspergrid_y = int(cp.ceil(ROWS / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # i have the power of god and gpu on my side
    GROWTHS[GROWTH_FUNC][blockspergrid, threadsperblock](
        bufferMatx, convMatx, somK, cp.array((mu, sigma)))

    deltaT = .05
    # introcuces delta time and sums up everything then clips beteen o and 1
    nextMatx = matrix + (bufferMatx * deltaT)
    nextMatx = cp.clip(nextMatx, 0., 1.)

    # if program is not paused
    if not paused:
        # saves prevoious state in memory
        prevMat = cp.copy(matrix)
        # updates matrix
        matrix = cp.copy(nextMatx)

        if forwOnes:  # handles taking only one step
            forwOnes = not forwOnes
            paused = True

    # renders the matrix
    render(matrix)

    if p:
        print('[NEXT MATRIX]\n', nextMatx)
        print('[CONV]\n', convMatx)
        print('[BUFFER]\n', bufferMatx)
        # print('[KERNEL]\n', kernel)
        p = False
