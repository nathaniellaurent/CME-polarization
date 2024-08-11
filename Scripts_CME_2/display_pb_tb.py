from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import mplcursors
from helper_funcs import *


fits_file_pB = fits.open('CME_1_pB\stepnum_065.fits')
fits_file_tB = fits.open('CME_1_tB\stepnum_065.fits')
fits_file_pB2 = fits.open('CME_0_pB\stepnum_020.fits')
fits_file_tB2 = fits.open('CME_0_tB\stepnum_020.fits')

image_data_pB = fits_file_pB[0].data 
image_data_tB = fits_file_tB[0].data
image_data_pB2 = fits_file_pB2[0].data
image_data_tB2 = fits_file_tB2[0].data


totalB = 0
height, width = image_data_pB.shape
for i in range(height):
    for j in range(width):
        totalB += image_data_pB[i][j]
print("total: ", totalB)


# print(image_data_pB)
# print(image_data_tB)
print("pB: ", image_data_pB[518][695])
print("pB sub: ", image_data_pB[519][730])
print("tB: ",image_data_tB[518][695])
print("tB sub: ",image_data_tB[519][730])

plt.figure()
plt.imshow(image_data_tB, origin='lower', norm=LogNorm())
plt.colorbar()
# plt.show()
plt.figure()
plt.imshow(image_data_pB, origin='lower', norm=LogNorm())
plt.colorbar()

plt.show()