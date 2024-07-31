from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import mplcursors
from helper_funcs import *


fits_file_pB = fits.open('CME_0_pB\stepnum_078.fits')
fits_file_tB = fits.open('CME_0_tB\stepnum_078.fits')
fits_file_pB2 = fits.open('CME_0_pB\stepnum_020.fits')
fits_file_tB2 = fits.open('CME_0_tB\stepnum_020.fits')

image_data_pB = fits_file_pB[0].data
image_data_tB = fits_file_tB[0].data
image_data_pB2 = fits_file_pB2[0].data
image_data_tB2 = fits_file_tB2[0].data

# print(image_data_pB)
# print(image_data_tB)
print(image_data_pB[900][600])
print(image_data_tB[900][600])

plt.figure()
plt.imshow(image_data_pB, norm=LogNorm())
plt.colorbar()
# plt.show()
plt.figure()
plt.imshow(image_data_tB, norm=LogNorm())
plt.colorbar()





xMin = 500
xMax = 800

yMin = 350
yMax = 650

# xMin = 0
# xMax = 1023

# yMin = 0
# yMax = 1023

pBdata = []
tBdata =[]

for i in range(5,60,5):
    if i < 10:
        st = '0' + str(i)
    else:
        st = str(i)
    fits_file_pB = fits.open(f'CME_0_pB\stepnum_0{st}.fits')
    fits_file_tB = fits.open(f'CME_0_tB\stepnum_0{st}.fits')
    pBdata.append(fits_file_pB[0].data)
    tBdata.append(fits_file_tB[0].data)

posMats = []
negMats = []

for i in range(len(pBdata)):
    print("Processing subtraction for image ", i)
    # pBdata[i] = subtractRadialMedian(pBdata[i], calculate_median_pixel_values(pBdata[i]))   
    # tBdata[i] = subtractRadialMedian(tBdata[i], calculate_median_pixel_values(tBdata[i]))  
    # pBdata[i] = minSmooth(pBdata[i], 50)  
    # tBdata[i] = minSmooth(tBdata[i], 50)   
    pBdata[i] = pBdata[i] - pBdata[10]
    tBdata[i] = tBdata[i] - tBdata[10]

fig, axs = plt.subplots(2,len(pBdata),figsize=(15, 20), sharex=True, sharey=True)


for i in range(len(pBdata)): 
    axs[0,i].imshow(pBdata[i][yMin:yMax,xMin:xMax], origin='lower')
    im = axs[1,i].imshow(tBdata[i][yMin:yMax,xMin:xMax], origin='lower')

plt.show()