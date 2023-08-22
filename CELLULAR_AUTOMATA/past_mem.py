import os
import numpy as np
import time


def printArray(a):
    print('||', end='')
    for n in a:
        print(f' {n} |', end='')
    print('|')


def printHead(h):
    pos = np.zeros((arr.shape[0]))
    pos[h] = 1

    print('  ', end='')
    for index, n in enumerate(pos):
        if n == 1:
            if arr[h] <= 9:
                print(' ↑  ', end='')
            else:
                print(' ↑↑  ', end='')
        else:
            if arr[index] < 10:
                print('    ', end='')
            else:
                print('     ', end='')


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
headPos = -1
limit = 15


running = True
while running:

    nextNumb = arr[-1] + 1
    print('[NEXT] ', nextNumb)

    action = input('\n> ')

    if action == 'e':
        running = False

    elif action == 'a':
        headPos -= 1
        if arr.shape[0] + headPos < 0:
            headPos += 1

    elif action == 'd':
        if headPos < -1:
            headPos += 1
        else:
            if arr.shape[0] < 15:
                arr = np.append(arr, nextNumb)
            else:
                arr = np.roll(arr, -1)
                arr[-1] = nextNumb

    elif action == '':
        if headPos < -1:
            headPos += 1
        else:
            if arr.shape[0] < 15:
                arr = np.append(arr, nextNumb)
            else:
                arr = np.roll(arr, -1)
                arr[-1] = nextNumb

    os.system('cls')
    printArray(arr)
    printHead(headPos)
