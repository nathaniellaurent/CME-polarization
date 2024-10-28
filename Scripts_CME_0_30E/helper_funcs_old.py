import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

def calculateExitAngles(image_data_pB, image_data_tB, xConstraints, yConstraints, point, type = 'Xi' ):
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
                if(point[0] == i and point[1] == j):
                    epsilonPoint = epsilon
                
                
                if(image_data_tB[i][j] > 0):
                    pBratio = pBratioFull[i][j]
                    
                    if(type == 'Xi'):
                        angleMatrixPositive[i][j] = epsilon + np.rad2deg(np.arcsin(np.sqrt((1 - pBratio)/(1 + pBratio))))
                        angleMatrixNegative[i][j] = epsilon + np.rad2deg(np.arcsin(-np.sqrt((1 - pBratio)/(1 + pBratio))))
                    if(type == 'Chi'):
                        angleMatrixPositive[i][j] = np.rad2deg(np.arccos(np.sqrt((1 - pBratio)/(1 + pBratio))))
                        angleMatrixNegative[i][j] = np.rad2deg(np.arccos(-np.sqrt((1 - pBratio)/(1 + pBratio))))


            else: 
                angleMatrixPositive[i][j] = 0
                angleMatrixNegative[i][j] = 0
        
    return angleMatrixPositive, angleMatrixNegative, epsilonPoint


def inverseR3(x,a,b,p):
    
    return a/(x**p) + b

def functionFitSubtract(image_data, point, direction='right'):
    #input 'point' is the only point that will be subtracted from

    print(point)

    line = lambda x, a, b: a*x + b

    parameters, covariance = curve_fit(line, [point[1], 512], [point[0], 512])
   
   
    
    

    # if(direction == 'right'):
    #     median_values = image_data[512,512:1024]
        
    # if(direction == 'left'):
    #     median_values = image_data[512,0:512]
    # r_values = np.arange(0,512)
    
    if(direction == 'right'):
        x_values = np.arange(512,1024)
        
    if(direction == 'left'):
        x_values = np.arange(0,512)
    

    
    y_values = line(x_values, parameters[0], parameters[1])
    
   

    r_values = np.sqrt((y_values - 512)**2 + (x_values - 512)**2)
    over_512 = np.where(r_values > 512)
    r_values = np.delete(r_values, over_512)
    x_values = np.delete(x_values, over_512)
    y_values = np.delete(y_values, over_512)

    plt.figure()
    plt.plot(x_values,y_values)
    plt.plot(point[1],point[0],'ro')
    plt.show()
    median_values = image_data[y_values.astype(int),x_values.astype(int)]

    print("r_values: ", r_values)
    print("median_values: ", median_values)
    print("x_values: ", x_values)
    print("y_values: ", y_values)


    
    # Remove values within 100 of the second index (the goal of this is to not fit to the CME)

    print("ignoring: ", max(25,int(0.3*np.sqrt((point[1] - 512)**2 + (point[0]- 512)**2))))
    interval =  max(25,int(0.5*np.sqrt((point[1] - 512)**2 + (point[0]- 512)**2)))
    print("start: ", point[1] - 512 - interval)
    print("end: ", point[1] - 512 + interval)
    delete = np.where(np.logical_and(r_values > np.sqrt((point[1] - 512)**2 + (point[0]- 512)**2) - interval, r_values < np.sqrt((point[1] - 512)**2 + (point[0]- 512)**2) + interval))
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
    print("a: ", parameters[0], "b: ", parameters[1], "p: ", parameters[2])
   
    fit_y = inverseR3(r_values_old, *parameters)
    
    # print(fit_y)

    plt.figure()
    plt.plot(r_values,median_values)
    plt.plot(r_values_old,median_values_old)
    plt.plot(r_values_old,fit_y)
    plt.plot(np.sqrt((point[1] - 512)**2 + (point[0]- 512)**2), image_data[point[0],point[1]], 'ro')
    
    plt.show()

    #evaluate the function at the point and subtract it from the image data
    print("Subtracting ", inverseR3(np.sqrt((point[1] - 512)**2 + (point[0]- 512)**2), *parameters), " from ", image_data[point[0],point[1]])
    image_data[point[0],point[1]] = image_data[point[0],point[1]] - inverseR3(np.sqrt((point[1] - 512)**2 + (point[0]- 512)**2),*parameters)

    return image_data