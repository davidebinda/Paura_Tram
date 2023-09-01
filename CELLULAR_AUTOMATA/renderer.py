import pygame
import cupyx.scipy.signal as sn
import numpy as np
import cupy as cp
from numba import cuda
from rules import GROWTHS
from matplotlib import cm
from tost import rasterize
import random
from datetime import datetime

# SCREEN VARS
WIDTH = 1600
HEIGHT = 890
'''WIDTH = 500
HEIGHT = 500'''
SCREEN_REALESTATE = (WIDTH, HEIGHT)
RES = 5
COLS = int(WIDTH/RES)
ROWS = int(HEIGHT/RES)
V_RES = RES
V_COLS = int(WIDTH/V_RES)
V_ROWS = int(HEIGHT/V_RES)

# SIMULATION VARS
simVar = {
    'rule': '',  # solo regole speciali per GOL
    'kEq': '',  # decide quale eq rateriz. per kernel
    'gEq': '',  # decide quale equazione rasterizzare per growth
    'mu': .0,
    'sigma': .0,
    'kernelRadius': 0,
    'deltaT': .0
}


# OTHER STUFF
mouseStart = (0, 0)
mouseCurrent = (0, 0)

consoleInput = ''
consoleInputBackup = 'tumadre'
consoleColor = (220, 220, 220)
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


# loads ssv | mode: k = kernel, m = matrix
def loadSSV(file=None, rad=None, mode=None):

    if mode == 'r':
        return rasterize(int(rad))  # create kernel from continuos equation

    values = []

    with open(file, "r") as f:

        global simVar

        lines = f.readlines()

        # gets all variables in SSV
        args = lines[0].split(',')
        args = [x.strip() for x in args]
        # print('\n\n', args, '\n\n')

        # gets height and width
        width = int(args[0].split(' ')[-1])
        height = int(args[1].split(' ')[-1])

        # gets the rest
        args = args[2:]
        for a in args:

            tag = a.split(' = ')[0]
            value = a.split(' = ')[-1]
            try:
                value = float(value)  # converts value to float if its a number
            except:
                pass

            simVar[tag] = value

        # sets deltaTime for rule B3/S23 since RLEs dont have it
        if simVar['rule'] == 'B3/S23':
            simVar['deltaT'] = 1.
            simVar['kEq'] = 'GOL'
            simVar['gEq'] = 'step'

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


# sprinkles magic squares to generate life from nothingness
def sprinkleMatx(r):
    m = cp.zeros((COLS, ROWS))
    r = int(r)
    randArea = cp.random.random((np.clip(r+2, 1, COLS), np.clip(r+2, 1, ROWS)))

    # print(f'[M] {m.shape[0]}x{m.shape[1]}')
    # print(f'[randArea] {randArea.shape[0]}x{randArea.shape[1]}')

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

        '''print(f'  {up}')
        print(f'{left}  {right}')
        print(f'  {down}')'''

        m[left:right, up:down] = randArea

    return m


def convert(args):
    for i in range(len(args)):
        # print(args[i])
        if len(args[i]) == 1:
            args[i] = args[i][0]
        elif len(args[i]) == 2:
            print(args[i])
            if args[i][0] == 'f':
                args[i] = float(args[i][1])
            elif args[i][0] == 'i':
                x = float(args[i][1])
                args[i] = int(x)
            elif args[i][0] == 'b':
                if args[i][1] == 'True' or args[i][1] == 'true' or args[i][1] == 't':
                    args[i] = True
                elif args[i][1] == 'False' or args[i][1] == 'false' or args[i][1] == 'f':
                    args[i] = False
        else:
            args[i] = str(args[i])
    return args


def clear():
    global matrix
    matrix = cp.zeros((COLS, ROWS))
    return 'Hampter'


# updates files record
def updateFilesRecord():
    files = []
    with open('SSV/RLEs/record.txt', 'r') as f:
        files = f.readlines()

    files = [x.strip('\n') for x in files]
    return files


# it saves to file ssv a matrix
def save(m, values):
    timeStamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    fileName = 'zzz_' + values['rule'] + '[' + timeStamp + ']'
    fileName = fileName.replace(':', '=')
    print('[SAVED TO]', fileName)

    stringToSave = ''

    # ELIMINA RIGHE E COLONNE VUOTE dove sum() = 0
    m = np.array(m.get())

    matx = np.ones((m.shape[1], m.shape[0]))
    for i in range(m.shape[0]):
        matx[:, i] = m[i, :]

    m = matx

    m = m[(m != 0).sum(1) != 0]
    column_sums = np.sum(m, axis=0)
    m = m[:, column_sums != 0]

    # adds dimentions
    stringToSave += 'x = ' + str(m.shape[1]) + ', '
    stringToSave += 'y = ' + str(m.shape[0]) + ', '

    # adds parameters
    for v in simVar.keys():
        stringToSave += v + ' = ' + str(values[v]) + ', '
    stringToSave = stringToSave[:-2] + '\n'

    for l in m:
        stringToSave += '0.0 '  # this is needed because im stupid
        for v in l:
            stringToSave += str(v) + ' '
        stringToSave = stringToSave + "\n"

    # creates file
    with open('SSV/' + fileName+'.ssv', 'w') as f:
        f.write(stringToSave)

    with open('SSV/RLEs/record.txt', 'a') as f:
        f.write('\n' + fileName+'.rle')


# it's terminal time
def terminal():
    global consoleInput
    global consoleInputBackup
    global consoleColor
    bMes = ''  # backMessage

    if len(consoleInput) > 0:
        if consoleInput[-1] == ')':

            consoleInput = consoleInput[:-1]
            consoleInputBackup = consoleInput

            actionMode = consoleInput.split('/')[0]
            consoleInput = consoleInput[2:]
            # print(actionMode)

            if actionMode == 'v':  # handles variables SEEMS TO WORK
                var = ''
                value = ''

                if '=' in consoleInput:
                    var = consoleInput.split('=')[0].strip()
                    value = consoleInput.split('=')[1].strip()
                else:
                    var = consoleInput

                # se value è un array:

                if var in globals().keys():
                    if value != '':
                        value = value.split(':')
                        print('[V] ', value)
                        globals()[var] = convert([value])[0]
                        bMes = globals()[var]
                    else:
                        bMes = globals()[var]

                    consoleColor = (0, 220, 0)

                else:
                    bMes = var + " doesn't exist"
                    consoleColor = (220, 0, 0)

                print('[VALUE] ', value)
                consoleInput = var + ' = ' + str(bMes)

            elif actionMode == 'f':  # handles functions KINDA WORKS KINDA
                var = ''
                if '=' in consoleInput:
                    var = consoleInput.split('=')[0].strip()
                    consoleInput = consoleInput.split('=')[1].strip()

                given = consoleInput.split('(')
                func = given[0]
                args = None
                print(func, end='\t')

                if len(given) > 1:
                    args = given[1]
                    args = [c.replace(' ', '') for c in args.split(',')]
                    args = [c for c in args if c != '']
                    args = [c.split(':') for c in args]

                args = convert(args)
                print(args)

                try:
                    if var != '' and var in globals().keys():
                        if args == None:
                            print('APPLY VARIABLE')
                            globals()[var] = globals()[func]()
                        else:
                            globals()[var] = globals()[func](*args)

                        bMes = var + ' = ' + str(globals()[var])
                    else:
                        if args == None:
                            bMes = globals()[func]()
                        else:
                            bMes = globals()[func](*args)

                    # colors output grren
                    consoleColor = (0, 220, 0)

                except Exception as e:
                    bMes = e
                    consoleColor = (220, 0, 0)

                consoleInput = str(bMes)

            elif actionMode == 'd':  # handles dictionaries
                var = ''
                value = ''

                if '=' in consoleInput:
                    var = consoleInput.split('=')[0].strip()
                    value = consoleInput.split('=')[1].strip()
                else:
                    var = consoleInput

                if var in globals()['simVar'].keys():
                    if value != '':
                        value = value.split(':')
                        print('[D] ', value)
                        globals()['simVar'][var] = convert([value])[0]
                        bMes = globals()['simVar'][var]
                    else:
                        bMes = globals()['simVar'][var]
                    consoleColor = (0, 220, 0)

                else:
                    bMes = var + " doesn't exist in simVar"
                    consoleColor = (220, 0, 0)

                print('[FROM DICT] ', value)
                consoleInput = var + ' = ' + str(bMes)

            else:
                bMes = '[VALID MODE NEEDED]'
                consoleColor = (220, 0, 0)
                consoleInput = str(bMes)

            # print(bMes)

    else:
        consoleColor = (220, 220, 220)

    s = pygame.Surface((WIDTH, 45), pygame.SRCALPHA)   # per-pixel alpha
    # notice the alpha value in the color
    s.fill((0, 0, 0, 150))
    screen.blit(s, (0, 0))

    console_surface = font.render(
        '~TERMINL~ '+consoleInput, True, consoleColor)
    screen.blit(console_surface, (10, 10))


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
    screen.blit(text_surface, (WIDTH - text_surface.get_width()-20,
                HEIGHT - text_surface.get_height()-10))

    # renders selection rectangle
    if rectMode:

        selecRect = pygame.Rect(mouseStart[0], mouseStart[1], 100, 100)
        selecColor = pygame.Color(255, 255, 0, 120)
        pygame.draw.rect(screen, selecColor, selecRect, 3)

    # calls terminal
    if terminalMode:
        terminal()

    fps_counter()

    pygame.display.flip()
    clock.tick(FRAMES)


#####################################################################
# by default loads last SSV added to record
files = updateFilesRecord()
filesIndex = -1
file = files[filesIndex][:-4]

print('[SIMULATING LAST ADDED TO RECORD]->', file)


# loads default SSV file
matrix = loadSSV(file="SSV/" + file + ".ssv", mode='m')
# KERNEL BE LIKE (not used in lenia mode)
kernel, somK = loadSSV(rad=simVar['kernelRadius'], mode='r')

#########################################

# OMG THE LOOP WOW SO COOL
prevMat = cp.copy(matrix)
p = True
running = True
paused = True
forwOnes = False
terminalMode = False
rectMode = False

while running:
    # gets inputs
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
    GROWTHS[simVar['rule']][blockspergrid, threadsperblock](
        bufferMatx, convMatx, somK, cp.array((simVar['mu'], simVar['sigma'])))

    # print(simVar)
    # simVar['rule']

    # introcuces delta time and sums up everything then clips beteen o and 1
    nextMatx = matrix + (bufferMatx * simVar['deltaT'])
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

    if True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # keyboard inputs
            if event.type == pygame.KEYDOWN:
                # handles terminal input

                if event.key == pygame.K_ESCAPE:
                    terminalMode = not terminalMode

                elif terminalMode:
                    # handles text input
                    if event.key == pygame.K_BACKSPACE:
                        consoleInput = consoleInput[:-1]
                    elif event.key == pygame.K_UP:
                        consoleInput = consoleInputBackup
                    elif event.key == pygame.K_DELETE:
                        consoleInput = ''
                    elif event.key == pygame.K_RETURN:
                        consoleInput += ')'
                    else:
                        consoleInput += event.unicode

                    break

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
                    matrix = sprinkleMatx(simVar['kernelRadius'])
                    file = ''
                elif event.key == pygame.K_DOWN:
                    # loads next SSV from record
                    files = updateFilesRecord()
                    filesIndex = (filesIndex + 1) % len(files)
                    matrix = loadSSV(
                        "SSV/" + files[filesIndex][:-4] + ".ssv", mode='m')
                    file = files[filesIndex][:-4]
                    pastMemories = []
                    pastIndex = -1
                    print(simVar)

                elif event.key == pygame.K_UP:
                    # loads next SSV from record
                    files = updateFilesRecord()
                    filesIndex = (filesIndex - 1) % len(files)
                    matrix = loadSSV(
                        "SSV/" + files[filesIndex][:-4] + ".ssv", mode='m')
                    file = files[filesIndex][:-4]
                    pastMemories = []
                    pastIndex = -1
                    print(simVar)

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

                elif event.key == pygame.K_p:
                    save(matrix, simVar)

            # mouse inputs
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    rectMode = True
                    mouseStart = pygame.mouse.get_pos()
                    mouseCurrent = mouseStart

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    rectMode = False

        # movement handler and others
        keys = pygame.key.get_pressed()
        if not terminalMode:
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
        else:
            # backspace for terminal input
            if keys[pygame.K_BACKSPACE] == True and keys[pygame.K_LCTRL] == True:
                consoleInput = consoleInput[:-1]

    if p:
        print('[NEXT MATRIX]\n', nextMatx)
        print('[CONV]\n', convMatx)
        print('[BUFFER]\n', bufferMatx)
        # print('[KERNEL]\n', kernel)
        p = False
