import pygame
import cupy as cp
from matplotlib import cm
import numpy as np


# Initializing Pygame
pygame.init()

# Initializing surface
surface = pygame.display.set_mode((400, 300))

# colors
COLOR_PRECISION = 100000
colors = cm.jet(np.linspace(0, 1, COLOR_PRECISION))
colors = [c*255 for c in colors]


def jet_colormap(value):
    return cm.jet(value)


x = 0
running = True
while running:
    print(x)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Changing surface color

    index = int(x * COLOR_PRECISION)
    # print(col[index])
    surface.fill(colors[index])
    pygame.display.flip()

    x = (x + 0.0001) % 1

'''    import numpy as np
import matplotlib.pylab as pl

x = np.linspace(0, 2*np.pi, 64)
y = np.cos(x) 

pl.figure()
pl.plot(x,y)

n = 20
colors = pl.cm.jet(np.linspace(0,1,n))

for i in range(n):
    pl.plot(x, i*y, color=colors[i])'''
