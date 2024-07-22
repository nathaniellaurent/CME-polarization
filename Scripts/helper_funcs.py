import numpy as np

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
    print(angleMatrixPositive[0][0])
    pBratioFull = image_data_pB/image_data_tB
    # plt.figure()
    # plt.imshow(pBratioFull, origin='lower', norm=LogNorm())
    # plt.colorbar()
    for i in range(yMin,yMax):
        for j in range(xMin,xMax):
            y = np.abs(halfY - i)
            y = y*45/halfY

            x = np.abs(halfX- j)
            x = x*45/halfX
            
            epsilon = np.sqrt(x*x + y*y)
            # print(epsilon)
            if(image_data_tB[i][j] > 0):
                pBratio = pBratioFull[i][j]
                
                angleMatrixPositive[i][j] = epsilon + np.rad2deg(np.arcsin(np.sqrt((1 - pBratio)/(1 + pBratio))))
                angleMatrixNegative[i][j] = epsilon + np.rad2deg(np.arcsin(-np.sqrt((1 - pBratio)/(1 + pBratio))))

    print("positive: ", angleMatrixPositive[512][645])
    print("negative: ", angleMatrixNegative[512][645])
    return angleMatrixPositive, angleMatrixNegative

def calculate_median_pixel_values(image_data):
    # Get the dimensions of the image
    height, width = image_data.shape

    # Calculate the center coordinates
    center_x = width // 2
    center_y = height // 2

    # Calculate the maximum radius
    max_radius = int(np.sqrt(center_x**2 + center_y **2)) +1 

    # Initialize an empty list to store the median pixel values
    median_values = []

    # Iterate over the different bands of radius
    for radius in range(5, max_radius + 1, 5):
        
        # Calculate the indices of the pixels within the current band of radius
        greater = np.sqrt((np.arange(height)[:, np.newaxis] - center_y) ** 2 +
                          (np.arange(width) - center_x) ** 2) <= radius
        # plt.figure()
        # plt.imshow(greater)
        less = np.sqrt((np.arange(height)[:, np.newaxis] - center_y) ** 2 +
                       (np.arange(width) - center_x) ** 2) > radius - 5
        
        sides = (np.arctan(np.abs(np.arange(height)[:, np.newaxis] - center_y)/
                           np.abs(np.arange(width) - center_x))) < (90-22.5)*np.pi/180
        # print(sides)

        greaterLess = np.logical_and(greater, less)
        both = np.logical_and(greaterLess, sides)  

        # if(radius%500 == 0):
        #     plt.figure()
        #     plt.imshow(both)
        
        # print(greater)
        # print(less)

        indices = np.where(both)
        # print(indices)

        # Extract the pixel values within the current band of radius
        pixels = image_data[indices]

        # Calculate the median pixel value
        median_value = np.quantile(pixels,0.5)

        # Append the median value to the list
        median_values.append(median_value)

    return median_values

def subtractRadialMedian(image_data, median_values):
    height, width = image_data.shape
    imageSubtract = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            radius = np.sqrt((i - height//2)**2 + (j - width//2)**2)
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
    # plt.imshow(image_data - imageSubtract, origin='lower', norm=LogNorm())

    return image_data - imageSubtract