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
plt.imshow(image_data_pB, origin='lower', norm=LogNorm())
plt.colorbar()
# plt.show()
plt.figure()
plt.imshow(image_data_tB, origin='lower', norm=LogNorm())
plt.colorbar()


# front_array = [[0,0],[0,0],[108,159],[140,160],[171,163],[195,175],[216,185],[241,167],[259,183],[274,172],[0,0]]
front_array = [[0,0],[0,0],[108,159],[140,160],[171,163],[195,175],[230,169],[261,167],[283,173],[0,0],[0,0]]
front_array = np.array(front_array)



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

posMats = np.empty((len(pBdata),len(pBdata[0]),len(pBdata[0][0])))
negMats = np.empty((len(pBdata),len(pBdata[0]),len(pBdata[0][0])))

for i in range(len(pBdata)):
    print("Processing subtraction for image ", i)
    # pBdata[i] = subtractRadialMedian(pBdata[i], calculate_median_pixel_values(pBdata[i]))   
    # tBdata[i] = subtractRadialMedian(tBdata[i], calculate_median_pixel_values(tBdata[i]))  
    pBdata[i] = minSmooth(pBdata[i], 50)  
    tBdata[i] = minSmooth(tBdata[i], 50)   
    

for i in range(len(pBdata)):
    print("Processing calculations for image ", i)
    posMat, negMat = calculateExitAngles(pBdata[i], tBdata[i], (xMin,xMax), (yMin,yMax))
    posMats[i] = posMat
    negMats[i] = negMat



vmin = -20
vmax = 80

# plt.figure()
# plt.imshow(image_data_pB, origin = 'lower', norm = LogNorm())
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(image_data_tB, origin = 'lower', norm = LogNorm())
# plt.colorbar()
# plt.show()


fig, axs = plt.subplots(3,len(pBdata),figsize=(15, 20), sharex=True, sharey=True)


for i in range(len(pBdata)): 
    axs[0,i].imshow(posMats[i][yMin:yMax,xMin:xMax], origin='lower', vmin=vmin, vmax=vmax, cmap='hsv')
    axs[1,i].imshow(pBdata[i][yMin:yMax,xMin:xMax], origin='lower', norm=LogNorm())
    im = axs[2,i].imshow(negMats[i][yMin:yMax,xMin:xMax], origin='lower', vmin=vmin, vmax=vmax, cmap='hsv')


# fig, axs = plt.subplots(2,6,figsize=(15, 20), sharex=True, sharey=True)


# for i in range(6):
#     axs[0,i].imshow(negMats[i][yMin:yMax,xMin:xMax], origin='lower', vmin=vmin, vmax=vmax, cmap='hsv')
#     if i < 5:
#         im = axs[1,i].imshow(negMats[i+6][yMin:yMax,xMin:xMax], origin='lower', vmin=vmin, vmax=vmax, cmap='hsv')
    

# Add data cursor to the plot
# print(posMats)
posMatsData = posMats[:,yMin:yMax,xMin:xMax]
negMatsData = negMats[:,yMin:yMax,xMin:xMax]
posMatsGraph = np.zeros(len(posMatsData))
negMatsGraph = np.zeros(len(negMatsData))


    

# Show the plot

fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

for i in range(len(posMatsData)):
    posMatsGraph[i] = posMatsData[i][front_array[i][1],front_array[i][0]]
    negMatsGraph[i] = negMatsData[i][front_array[i][1],front_array[i][0]]
print(posMatsGraph)
print(negMatsGraph)

plt.figure()   
plt.plot(posMatsGraph[2:9], label='Positive Exit Angle')
plt.plot(negMatsGraph[2:9], label='Negative Exit Angle') 

plt.show()