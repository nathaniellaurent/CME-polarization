from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import mplcursors
from helper_funcs import *
from scipy import signal
from scipy.optimize import curve_fit

maxImages = 50

xMin = 512
xMax = 1000



yMin = 250
yMax = 750


front_array_absolute = [[512,608],[512,640],[512,671],[512,702],[512,730],[512,761],[512,783]]
                                                           
front_array_absolute = np.array(front_array_absolute)

def linearFunc( x, m, b):
    return m*x + b

def inverseR3(x,a,b):
    
    return a/(x**3) + b

parameters, covariance = curve_fit(linearFunc, np.arange(15,maxImages,5), front_array_absolute[:,1])
parametersX, covarianceX = curve_fit(linearFunc, np.arange(15,maxImages,5), front_array_absolute[:,0])
front_array_absolute = np.zeros((maxImages-15,2), dtype=int)
front_array_absolute[:,0] = np.floor((linearFunc(np.arange(15,maxImages), *parametersX)))
front_array_absolute[:,1] = np.floor((linearFunc(np.arange(15,maxImages), *parameters)))

pBdata = []
tBdata =[]

for i in range(15,maxImages,1):
    if i < 10:
        st = '0' + str(i)
    else:
        st = str(i)
    fits_file_pB = fits.open(f'CME_0_pB\stepnum_0{st}.fits')
    fits_file_tB = fits.open(f'CME_0_tB\stepnum_0{st}.fits')
    pBdata.append(fits_file_pB[0].data)
    tBdata.append(fits_file_tB[0].data)

BRdata = []
BTdata = []
for i in range(len(pBdata)):
    BRdata.append(0.5*( tBdata[i] -pBdata[i] ))
    BTdata.append(0.5*( tBdata[i] + pBdata[i] ))


plt.figure()
plt.imshow(BTdata[15], norm=LogNorm())
plt.colorbar()
plt.figure()
plt.imshow(BRdata[15], norm=LogNorm())
plt.colorbar()

posMats = np.empty((len(pBdata),len(pBdata[0]),len(pBdata[0][0])))
negMats = np.empty((len(pBdata),len(pBdata[0]),len(pBdata[0][0])))


epsilonArray = np.zeros(len(pBdata))

i = 15


print("Processing subtraction for image ", i)

            

BRdata[i], parameters = functionFitSubtract(BRdata[i], front_array_absolute[i,:], 'right')
BTdata[i], parameters = functionFitSubtract(BTdata[i], front_array_absolute[i,:], 'right')

plt.figure()
plt.imshow(BTdata[15], norm=LogNorm(vmin=1e-9,vmax=1e-4), cmap='hot',extent=[-45,45, -45 , 45])
plt.xlabel('y (deg)')
plt.ylabel('x (deg)')

plt.colorbar(label='Pixel Brightness (B)')


for row in range(len(BRdata[i])):
    for col in range(len(BRdata[i][0])):
       
        radius = np.sqrt((col - 512)**2 + (row - 512)**2)
        # print("Value before: ", BRdata[15][row][col])
        # print("Subtract value: ", inverseR3(radius, *parameters))
        # print("Value after: ", BRdata[15][row][col] - inverseR3(radius, *parameters))
        BTdata[15][row][col] -= inverseR3(radius, *parameters)
        if(BTdata[15][row][col] < 0):
            BTdata[15][row][col] = 1e-9



plt.figure()
x = [0, 50]  # x-coordinates of the line
y = [0, 0]  # y-coordinates of the line
plt.plot(x, y, color='white', linestyle = '--',linewidth=2)

plt.imshow(BTdata[15], norm=LogNorm(vmin=1e-9,vmax=1e-4), cmap='hot',extent=[-45,45, -45 , 45])


plt.xlabel('y (deg)')
plt.ylabel('x (deg)')

plt.colorbar(label='Pixel Brightness (B)')
plt.show()
# Calculate the exit angles for each pixel in the region of interest
