import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

def calculateExitAngles(image_data_pB, image_data_tB, xConstraints, yConstraints ):
    xMin, xMax = xConstraints
    yMin, yMax = yConstraints
    angleMatrixPositive = np.zeros((len(image_data_pB),len(image_data_pB[0]) ))
    angleMatrixNegative = np.zeros((len(image_data_pB),len(image_data_pB[0]) ))

    # angleMatrixPositive = [[0] * len(image_data_pB[0])  for i in range(len(image_data_pB))]
    # angleMatrixNegative = [[0] * len(image_data_pB[0])  for i in range(len(image_data_pB))]

    Y = len(image_data_pB)
    halfY = Y/2.0
    X = len(image_data_pB[0])
    halfX = X/2.0
    pBratioFull = image_data_pB/image_data_tB
    # plt.figure()
    # plt.imshow(pBratioFull, origin='lower', norm=LogNorm())
    # plt.colorbar()
    for i in range(1024):
        
        # if(i%50 == 0):
        #     print(int(100*(i-yMin)/(yMax-yMin)), "% done")
        for j in range(1024):
            if( i < yMax and i >= yMin and j < xMax and j >= xMin):
                y = np.abs(halfY - i)
                y = y*45/halfY

                x = np.abs(halfX- j)
                x = x*45/halfX
                
                epsilon = np.sqrt(x*x + y*y)
                
                
                if(image_data_tB[i][j] > 0):
                    pBratio = pBratioFull[i][j]
                    # print("using arccos")
                    
                    angleMatrixPositive[i][j] = epsilon + np.rad2deg(np.arcsin(np.sqrt((1 - pBratio)/(1 + pBratio))))
                    angleMatrixNegative[i][j] = epsilon + np.rad2deg(np.arcsin(-np.sqrt((1 - pBratio)/(1 + pBratio))))
                    # angleMatrixPositive[i][j] = np.rad2deg(np.arccos(np.sqrt((1 - pBratio)/(1 + pBratio))))
                    # angleMatrixNegative[i][j] = np.rad2deg(np.arccos(-np.sqrt((1 - pBratio)/(1 + pBratio))))
            else: 
                angleMatrixPositive[i][j] = 0
                angleMatrixNegative[i][j] = 0
        
    return angleMatrixPositive, angleMatrixNegative

def calculateRadialBands(image_data, direction='right'):
    # Get the dimensions of the image
    height, width = image_data.shape

    # Calculate the center coordinates
    center_x = width // 2
    center_y = height // 2

    # Calculate the maximum radius
    max_radius = int(np.sqrt(center_x**2 + center_y **2)) +1 

    # Initialize an empty list to store the median pixel values
    
    allIndices = []
    # Iterate over the different bands of radius
    for radius in range(5, max_radius + 1, 5):
        
        # Calculate the indices of the pixels within the current band of radius
        greater = np.sqrt((np.arange(height)[:, np.newaxis] - center_y) ** 2 +
                          (np.arange(width) - center_x) ** 2) < radius
        # plt.figure()
        # plt.imshow(greater)
        less = np.sqrt((np.arange(height)[:, np.newaxis] - center_y) ** 2 +
                       (np.arange(width) - center_x) ** 2) >= radius - 5
        
        sides = (np.arctan(np.abs(np.arange(height)[:, np.newaxis] - center_y)/
                           np.abs(np.arange(width) - center_x))) < (90-22.5)*np.pi/180
        
        if(direction == 'left'):
            right = (0*(np.arange(height)[:, np.newaxis] - center_y)  +
                       (np.arange(width) - center_x)) < 0
        elif(direction == 'right'):
            right = (0*(np.arange(height)[:, np.newaxis] - center_y)  +
                       (np.arange(width) - center_x)) > 0
        # print(sides)

        greaterLess = np.logical_and(greater, less)
        both = np.logical_and(greaterLess, sides)  
        both = np.logical_and(both, right)

        if(radius%500 == 0):
            plt.figure()
            plt.imshow(both)
            plt.show()
        
        # print(greater)
        # print(less)

        indices = np.where(both)
        allIndices.append(indices)
        # print(indices)

        # Extract the pixel values within the current band of radius
        
    
    return allIndices
def calculateMedianPixelValues(image_data, allIndices):
    median_values = np.zeros(len(allIndices))
    
    

    for index in range(len(allIndices)):
        
        # print(indices)
        pixels = image_data[allIndices[index]]
        
        
        # Calculate the median pixel value
        median_value = np.quantile(pixels,0.5)
        # median_value = np.mean(pixels)

        # Append the median value to the list
        median_values[index] = median_value
    # print(median_values)

    # print(len(median_values))
    r_values = np.array(range(5, len(median_values)*5 + 1, 5))
    # print(len(r_values))
    return median_values, r_values

def subtractRadialMedian(image_data, median_values):
    # print(median_values)
    height, width = image_data.shape
    imageSubtract = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            radius = np.sqrt((i - height//2)**2 + (j - width//2)**2)
            if(radius < 55):
                continue
            lower = int(radius//5)
            upper = lower + 1
            if upper < len(median_values):
                lower_median = median_values[lower]
                upper_median = median_values[upper]
                weight = (radius/5) - lower
                imageSubtract[i][j] = (1-weight)*lower_median + weight*upper_median
            else:
                imageSubtract[i][j] = median_values[-1]
    # plt.figure()
    # plt.imshow(imageSubtract, origin='lower', norm=LogNorm())
    # plt.show()

    return image_data - imageSubtract


def minSmooth(image_data, kernel_size):
    for i in range(len(image_data)):
        for j in range(len(image_data[0])):
            if(image_data[i][j] == 0):
                image_data[i][j] = 100
    height, width = image_data.shape
    imageSubtract = np.zeros((height, width))
    for i in range(height):
        if(i%100 == 0):
            print(int(100*i/height), "% done")
        for j in range(width):
            subImage = image_data[max(0,i - kernel_size//2):min(height-1,i+kernel_size//2),max(0,j - kernel_size//2):min(width-1,j+kernel_size//2)]
            minVal = np.amin(subImage)
            imageSubtract[i][j] = minVal
    for i in range(len(image_data)):
        for j in range(len(image_data[0])):
            if(image_data[i][j] == 100):
                image_data[i][j] = 0
    # plt.figure()
    # plt.imshow(imageSubtract, origin='lower', norm=LogNorm())

    return image_data - imageSubtract


def nearbyPointSubtract(pBImage, tBImage, point, pixelDistance, allIndices, direction='right'):

    

    pointSub = []
    pointSub.append(point[0])
    if(direction == 'right'):
        pointSub.append(point[1] + pixelDistance)
    elif(direction == 'left'):
        pointSub.append(point[1] - pixelDistance)
    
    subValuepB = pBImage[pointSub[0],pointSub[1]]
    subValuetB = tBImage[pointSub[0],pointSub[1]]
    weight = (abs((pointSub[1] - 512)) - abs((pointSub[1] - 512))//5*5)/5.0

    radialFuncpB = calculateMedianPixelValues(pBImage,allIndices)
    radialFunctB = calculateMedianPixelValues(tBImage,allIndices)
    plt.figure()
    plt.plot(radialFuncpB)
    plt.show()
    
    print("radialFuncpB: ", radialFuncpB)
    print("radialFunctB: ", radialFunctB)

    print("point: ", point)
    print("subValuepB: ", subValuepB) 
    print("subValuetB: ", subValuetB)
    idxpB = (np.abs(radialFuncpB - subValuepB)).argmin()
    idxtB = (np.abs(radialFunctB - subValuetB)).argmin()

    if(radialFunctB[idxtB] > subValuetB):
        idxpBup = idxpB + 1
        idxtBup = idxtB + 1
    else:
        idxpBup = idxpB 
        idxtBup = idxtB
        idxpB = idxpB - 1
        idxtB = idxtB - 1

    # print(idxpB)
    # print(idxpB - pixelDistance//5)
    # print(radialFuncpB[idxpB - pixelDistance//5])
    # print(radialFunctB[idxtB - pixelDistance//5])
    
    print("prev: subtracting pB: ", pBImage[pointSub[0],pointSub[1]],  " from ", pBImage[point[0],point[1]] )
    print("actu: subtracting pB: ", (radialFuncpB[idxpB - pixelDistance//5]*(1-weight) + radialFuncpB[idxpBup - pixelDistance//5]*(weight)),  " from ", pBImage[point[0],point[1]] )
    print("prev: subtracting tB: ", tBImage[pointSub[0],pointSub[1]],  " from ", tBImage[point[0],point[1]] )
    print("actu: subtracting tB: ", (radialFunctB[idxtB - pixelDistance//5]*(1-weight) + radialFunctB[idxtBup - pixelDistance//5]*(weight)),  " from ", tBImage[point[0],point[1]] )
    # pBImage[point[0], point[1]] = pBImage[point[0],point[1]] - (radialFuncpB[idxpB - pixelDistance//5]*(1-weight) + radialFuncpB[idxpBup - pixelDistance//5]*(weight))
    # tBImage[point[0], point[1]] = tBImage[point[0],point[1]] - (radialFunctB[idxtB - pixelDistance//5]*(1-weight) + radialFunctB[idxtBup - pixelDistance//5]*(weight))
    pBImage[point[0], point[1]] = pBImage[point[0],point[1]] - radialFuncpB[idxpB - pixelDistance//5]
    tBImage[point[0], point[1]] = tBImage[point[0],point[1]] - radialFunctB[idxtB - pixelDistance//5]



    return pBImage, tBImage

def inverseR2(x,a,b):
    
    return a/(x**3) + b

def functionFitSubtract(image_data, point, direction='right'):
    print(point)
    if(direction == 'right'):
        median_values = image_data[512,512:1024]
        
    if(direction == 'left'):
        median_values = image_data[512,0:512]
    r_values = np.arange(0,512)
    

    #find first nonzero value of median_values
    

    # Find the second index of the point
    

    # Remove values within 100 of the second index
    print("ignoring: ", max(25,int(0.3*abs(point[1]-512))))
    interval =  max(25,int(0.3*abs(point[1]-512)))
    print("start: ", point[1] - 512 - interval)
    print("end: ", point[1] - 512 + interval)
    delete = np.arange(point[1] - 512 - interval, point[1] - 512 + interval)
    print(delete)
    
    r_values_old = r_values
    median_values_old = median_values

    plt.figure()
    plt.plot(r_values,median_values, 'o')
    r_values = np.delete(r_values, delete )
    median_values = np.delete(median_values, delete)
    
    plt.plot(r_values,median_values,'o')
    plt.show()

    if(direction == 'left'):
        median_values = np.flip(median_values)
    firstNonZero = 0
    for j in range(len(median_values)):
        if median_values[j] != 0:
            firstNonZero = j
            break

    median_values = median_values[firstNonZero+2:]
    r_values = r_values[firstNonZero+2:]

    firstNonZero_old = 0
    for j in range(len(median_values_old)):
        if median_values_old[j] != 0:
            firstNonZero_old = j
            break

    median_values_old = median_values_old[firstNonZero_old+2:]
    r_values_old = r_values_old[firstNonZero_old+2:]

   

    # median_values = median_values[0::5]
    # r_values = r_values[0::5]


    parameters, covariance = curve_fit(inverseR2, r_values, median_values)
    print("a: ", parameters[0], "b: ", parameters[1])
   
    fit_y = inverseR2(r_values_old, parameters[0], parameters[1])
    
    # print(fit_y)

    plt.figure()
    # plt.plot(r_values,median_values)
    plt.plot(r_values_old,median_values_old)
    plt.plot(r_values_old,fit_y)
    plt.plot(abs(point[1] - 512), image_data[point[0],point[1]], 'ro')
    
    plt.show()
    print("Subtracting ", inverseR2(abs(point[1] - 512), parameters[0], parameters[1]), " from ", image_data[point[0],point[1]])
    image_data[point[0],point[1]] = image_data[point[0],point[1]] - inverseR2(abs(point[1] - 512), parameters[0], parameters[1])
    return image_data
