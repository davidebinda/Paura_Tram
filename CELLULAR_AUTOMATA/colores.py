import pygame

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm

# Initializing Pygame
pygame.init()

# Initializing surface
surface = pygame.display.set_mode((400, 300))


# generate the data and plot it for an ideal normal curve

# x-axis for the plot
x_data = np.arange(0, 1, 0.0001)

# y-axis as the gaussian
'''muR = .05
sigmaR = .2
muG = .5
sigmaG = .2
muB = .9
sigmaB = .2


# Initializing RGB Color
def color(xxx):
    R = np.absolute(
        (np.clip(stats.norm.pdf(xxx, muR, sigmaR), 0, 1) - .5))
    G = np.absolute(
        (np.clip(stats.norm.pdf(xxx, muG, sigmaG), 0, 1) - .5))
    B = np.absolute(
        (np.clip(stats.norm.pdf(xxx, muB, sigmaB), 0, 1) - .5))
    return [R, G, B]


y_data_R, y_data_G, y_data_B = color(x_data)

plt.plot(x_data, y_data_R)
plt.plot(x_data, y_data_B)
plt.plot(x_data, y_data_G)
plt.show()'''


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
    col = [c*255 for c in jet_colormap(x)]
    print(col)
    surface.fill(col)
    pygame.display.flip()

    x = (x + 0.0001) % 1

'''import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def jet_colormap(value):
    return cm.jet(value)


# Esempio di utilizzo
values = np.linspace(0, 1, 100)  # Valori compresi tra 0 e 1
colors = [jet_colormap(value) for value in values]

plt.scatter(values, values, c=colors)
plt.show()'''
