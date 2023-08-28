'''import wget

URL = input('[RLE LINK] --> ')
fileName = input('[FILE NAME] --> ')

if fileName == '':
    response = wget.download(URL, 'SSV\RLEs')
else:
    response = wget.download(URL, 'SSV\RLEs\\'+fileName+'.rle')'''

import matplotlib.pyplot as plt
import numpy as np

mu, sigma = .5, 0.1  # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - mu)**2 / (2 * sigma**2)),
         linewidth=2, color='r')
plt.show()
