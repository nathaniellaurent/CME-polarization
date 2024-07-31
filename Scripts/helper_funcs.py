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


    Y = len(image_data_pB)
    halfY = Y/2.0
    X = len(image_data_pB[0])
    halfX = X/2.0
    pBratioFull = image_data_pB/image_data_tB

    for i in range(1024):
        
        for j in range(1024):
            if( i < yMax and i >= yMin and j < xMax and j >= xMin):
                y = np.abs(halfY - i)
                y = y*45/halfY

                x = np.abs(halfX- j)
                x = x*45/halfX
                
                epsilon = np.sqrt(x*x + y*y)
                
                
                if(image_data_tB[i][j] > 0):
                    pBratio = pBratioFull[i][j]
                    
                    angleMatrixPositive[i][j] = epsilon + np.rad2deg(np.arcsin(np.sqrt((1 - pBratio)/(1 + pBratio))))
                    angleMatrixNegative[i][j] = epsilon + np.rad2deg(np.arcsin(-np.sqrt((1 - pBratio)/(1 + pBratio))))
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

        greaterLess = np.logical_and(greater, less)
        both = np.logical_and(greaterLess, sides)  
        both = np.logical_and(both, right)

        #Show example band
        if(radius%500 == 0):
            plt.figure()
            plt.imshow(both)
            plt.show()
        

        indices = np.where(both)
        allIndices.append(indices)

        
    
    return allIndices

# Calculate the median pixel values for each band of radius provided by calculateRadialBands
def calculateMedianPixelValues(image_data, allIndices):
    median_values = np.zeros(len(allIndices))
    
    

    for index in range(len(allIndices)):
        
        pixels = image_data[allIndices[index]]
        
        # Calculate the median pixel value
        median_value = np.quantile(pixels,0.5)

        median_values[index] = median_value

    r_values = np.array(range(5, len(median_values)*5 + 1, 5))
    return median_values, r_values

# Subtract the median of each band of radius from the image data
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


def inverseR3(x,a,b):
    
    return a/(x**3) + b

def functionFitSubtract(image_data, point, direction='right'):
    #input 'point' is the only point that will be subtracted from

    print(point)


    if(direction == 'right'):
        median_values = image_data[512,512:1024]
        
    if(direction == 'left'):
        median_values = image_data[512,0:512]
    r_values = np.arange(0,512)
    
    # Remove values within 100 of the second index (the goal of this is to not fit to the CME)

    print("ignoring: ", max(25,int(0.3*abs(point[1]-512))))
    interval =  max(25,int(0.3*abs(point[1]-512)))
    print("start: ", point[1] - 512 - interval)
    print("end: ", point[1] - 512 + interval)
    delete = np.arange(point[1] - 512 - interval, point[1] - 512 + interval)
    # print(delete)
    
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
        median_values_old = np.flip(median_values_old)

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

   


    parameters, covariance = curve_fit(inverseR3, r_values, median_values)

    # print the parameters of the fit function a/(x^3) + b
    print("a: ", parameters[0], "b: ", parameters[1])
   
    fit_y = inverseR3(r_values_old, parameters[0], parameters[1])
    
    # print(fit_y)

    plt.figure()
    plt.plot(r_values,median_values)
    plt.plot(r_values_old,median_values_old)
    plt.plot(r_values_old,fit_y)
    plt.plot(abs(point[1] - 512), image_data[point[0],point[1]], 'ro')
    
    plt.show()

    #evaluate the function at the point and subtract it from the image data
    print("Subtracting ", inverseR3(abs(point[1] - 512), parameters[0], parameters[1]), " from ", image_data[point[0],point[1]])
    image_data[point[0],point[1]] = image_data[point[0],point[1]] - inverseR3(abs(point[1] - 512), parameters[0], parameters[1])

    return image_data


def medianOverTime(image_data_list):
    image_data_list = np.array(image_data_list)

    image_data_list = np.stack(image_data_list, axis=0)

    

    return np.quantile(image_data_list, 0.5, axis=0)
