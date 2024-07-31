from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import mplcursors
from helper_funcs import *

fits_file_pB = fits.open('CME_0_pB\stepnum_025.fits')
fits_file_tB = fits.open('CME_0_tB\stepnum_025.fits')
fits_file_pB2 = fits.open('CME_0_pB\stepnum_020.fits')
fits_file_tB2 = fits.open('CME_0_tB\stepnum_020.fits')

image_data_pB = fits_file_pB[0].data
image_data_tB = fits_file_tB[0].data
image_data_pB2 = fits_file_pB2[0].data
image_data_tB2 = fits_file_tB2[0].data

plt.figure()
plt.imshow(image_data_pB, origin='lower', norm=LogNorm())
plt.colorbar()

smoothImage = minSmooth(image_data_pB,50)
radialImage = subtractRadialMedian(image_data_pB, calculate_median_pixel_values(image_data_pB))
print("Initial: ",image_data_pB[776][205])
print("Smoothed: ", smoothImage[776][205])
print("Radial: ", radialImage[776][205])

plt.figure()
plt.imshow(radialImage, origin='lower', norm=LogNorm())
plt.colorbar()

plt.figure()
plt.imshow(smoothImage, origin='lower', norm=LogNorm())
plt.colorbar()

plt.show()