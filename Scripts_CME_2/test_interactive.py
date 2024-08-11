import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from astropy.io import fits
from matplotlib.colors import LogNorm


currentImageIdx = 10

# fits_tb = fits.open(f'CME_2_tB\stepnum_0{25}.fits')
# image_data_tb = fits_tb[0].data
# plt.figure()
# plt.imshow(image_data_tb, cmap="viridis", norm=LogNorm(), origin='lower')


# fits_pb = fits.open(f'CME_2_pB\stepnum_0{25}.fits')
# image_data_pb = fits_pb[0].data
# plt.figure()
# plt.imshow(image_data_pb, cmap="viridis", norm=LogNorm(), origin='lower')
# plt.show()


outputPoints = np.empty((7,2), dtype=float)
currentLine = None
currentImage = None

def getNextImage():
    global currentImageIdx, currentLine, currentImage
    currentImageIdx += 5

    if(currentImageIdx == 50):
        np.save("output/outputCenter.npy", outputCenter)
        np.save("output/outputRadius.npy", outputRadius)
        np.save("output/outputPoints.npy", outputPoints)
        exit()

    fits_file_pB = fits.open(f'CME_2_pB\stepnum_0{currentImageIdx}.fits')
    image_data_pB = fits_file_pB[0].data 
    if(currentImage != None):
        currentImage.remove()
    currentImage = ax.imshow(image_data_pB, cmap="viridis", norm=LogNorm(), origin='lower')
    
    # ax.plot(*outputPoints[0], color = 'white', marker = 'o')
    if(currentLine != None):
        currentLine[0].remove()
    currentLine = ax.plot([512,outputPoints[(currentImageIdx-20)//5][0]],[512,outputPoints[(currentImageIdx-20)//5][1]], color = 'white')
    # fig.canvas.draw()
    print((currentImageIdx-20)//5)
    print(outputPoints[0])
    
    


fig, ax = plt.subplots(constrained_layout=True)

klicker = clicker(ax, ["circles", "front"], markers=["x", "+"])


totalOutputs = 0

currentCircle = None
currentCenter = None
currentRadius = None
outputRadius = np.empty(7,dtype=float)
outputCenter = np.empty((7,2), dtype=float)

getNextImage()






def findCircle(p1, p2, p3) :
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    x12 = x1 - x2; 
    x13 = x1 - x3; 

    y12 = y1 - y2; 
    y13 = y1 - y3; 

    y31 = y3 - y1; 
    y21 = y2 - y1; 

    x31 = x3 - x1; 
    x21 = x2 - x1; 

    # x1^2 - x3^2 
    sx13 = pow(x1, 2) - pow(x3, 2); 

    # y1^2 - y3^2 
    sy13 = pow(y1, 2) - pow(y3, 2); 

    sx21 = pow(x2, 2) - pow(x1, 2); 
    sy21 = pow(y2, 2) - pow(y1, 2); 

    f = (((sx13) * (x12) + (sy13) * 
          (x12) + (sx21) * (x13) + 
          (sy21) * (x13)) // (2 * 
          ((y31) * (x12) - (y21) * (x13))));
            
    g = (((sx13) * (y12) + (sy13) * (y12) + 
          (sx21) * (y13) + (sy21) * (y13)) // 
          (2 * ((x31) * (y12) - (x21) * (y13)))); 

    c = (-pow(x1, 2) - pow(y1, 2) - 
         2 * g * x1 - 2 * f * y1); 

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0 
    # where centre is (h = -g, k = -f) and 
    # radius r as r^2 = h^2 + k^2 - c 
    h = -g; 
    k = -f; 
    sqr_of_r = h * h + k * k - c; 

    # r is the radius 
    r = round(np.sqrt(sqr_of_r), 5); 

    return (h,k), r
    

def point_added_cb(position: Tuple[float, float], klass: str):
    x, y = position
    if klass == "circles":
        global currentCircle, currentRadius, currentCenter, outputCenter, outputRadius, totalOutputs
        
        # points.append(position)
        # print(points)
        points = klicker.get_positions().get("circles")
        print(points)
        if(len(points) == 3):
            (centerx, centery) , radius = findCircle(points[0], points[1], points[2])   
            print(f"Circle with center {centerx}, {centery} and radius {radius}")
            # ax.plot(centerx, centery, 'ro')
            circle = plt.Circle((centerx, centery), radius, color='r', fill=False)
            currentCircle = circle

            ax.add_patch(circle)

            currentCenter = np.asarray([centerx, centery])
            currentRadius = radius

            fig.canvas.draw()
        if(len(points) == 4):

            currentCircle.remove()
            currentCircle = None

            outputCenter[totalOutputs] = currentCenter
            outputRadius[totalOutputs] = currentRadius

            currentRadius = None
            currentCenter = None

            klicker.clear_positions()
            getNextImage()

            totalOutputs += 1
            fig.canvas.draw()
    if klass == "front":
        global outputPoints
        outputPoints[totalOutputs] = np.asarray([x, y])

    

    print(f"New point of class {klass} added at {x=}, {y=}")


def point_removed_cb(position: Tuple[float, float], klass: str, idx):
    global currentCircle, currentRadius, currentCenter
    x, y = position
    
    if(len(klicker.get_positions().get("circles")) == 2):
        currentCircle.remove()
        currentCircle = None
        currentRadius = None
        currentCenter = None
        fig.canvas.draw()

    suffix = {'1': 'st', '2': 'nd', '3': 'rd'}.get(str(idx)[-1], 'th')
    print(
        f"The {idx}{suffix} point of class {klass} with position {x=:.2f}, {y=:.2f}  was removed"
    )


klicker.on_point_added(point_added_cb)
klicker.on_point_removed(point_removed_cb)

plt.show()




print(klicker.get_positions())