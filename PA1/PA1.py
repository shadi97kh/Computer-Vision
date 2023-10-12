# Import necessary libraries
from PIL import Image
from pylab import *
from copy import deepcopy
import cv2
import numpy as num, math
import numpy as np 
from math import pi, sqrt, exp
import matplotlib.pyplot as plt
from numpy import matrix
import matplotlib.cm as cm
from matplotlib import pyplot
import os

# Define Gaussian function
def Gaussian_Fn(n,sigma):
    
    # Generate list of values centered at zero
    size = range(-int(n/2),int(n/2)+1)
    # Return Gaussian values for each x in size
    return [1 / (sigma * sqrt(2*np.pi)) * exp(-float(x)**2/(2*sigma**2)) for x in size]


# Define Gaussian Derivative function
def Gaussian_Derrivative(n,sigma):
    size = range(-int(n/2),int(n/2)+1)
    # Return Gaussian derivative values for each x in size
    return [-x / (sigma**3*sqrt(2*np.pi)) * exp(-float(x)**2/(2*sigma**2)) for x in size]


# Save 2D numpy array data as an image file
def save_image(data, filename):
    im = Image.fromarray(np.uint8(data))
    im.save(filename)

#Implement Canny edge detection
def canny_edge_detection(I, sigma):
    # Convolve image with Gaussian along x-axis
    Ixg=[]
    g = Gaussian_Fn(7,sigma)
    for i in range(len(I[:,0])):
        x=np.convolve(I[i,:], g)
        Ixg.append(x)
    
    Ixg = np.array(np.matrix(Ixg))
    
    # Convolve image with Gaussian along y-axis
    Iyt =[]
    for i in range (len(I[0,:])):
        y = np.convolve(I[:,i], g)
        Iyt.append(y) 
    Iyg = np.transpose(Iyt)
    
    # Convolve Ixg with Gaussian derivative
    gd = Gaussian_Derrivative(7,8)
    Ix_gd = []
    for i in range(len(Ixg[:,0])):
        x = np.convolve(Ixg[i,:],gd)
        Ix_gd.append(x)
        
        
    # Convolve Ixg with Gaussian derivative
    IygdT=[]
    for i in range (len(Iyg[0,:])):
        y = np.convolve(Iyg[:,i], gd)
        IygdT.append(y) 
    Iy_gd= np.transpose(IygdT)

    Ix_gdsq= np.square(Ix_gd)
    Iy_gdsq= np.square(Iy_gd)

    # Calculate magnitude of gradient
    Magxy =[]
    for i in range (len(Ix_gdsq)):
        temp = []
        for j in range (len(Iy_gdsq[0,:])):
            temp.append(sqrt(Ix_gdsq[i,j] + Iy_gdsq[i,j]))
            if(j == len(Iy_gdsq[0,:])-1):
                Magxy.append(temp)
    Magxy = np.array(np.matrix(Magxy))


    # Calculate direction of gradient
    A= np.array(np.matrix(Ix_gd))
    B= np.array(np.matrix(Iy_gd))
    AngleDeg =[]
    for i in range(len(Ix_gd)):
        temp=[]
        for j in range(len(Iy_gd[0,:])):
            temp.append((math.atan2(B[i,j],A[i,j]))*180/pi)
            if(j == len(Iy_gd[0,:])-1):
                AngleDeg.append(temp)
                
    Angle= np.array(np.matrix(AngleDeg))
    
    # Apply non-maximum suppression
    Magxy_Temp = Magxy 
    NonMax  = deepcopy(Magxy)
    for i in range(len(Angle[:,0])):
        for j in range(len(Angle[0,:])):
            try:
                #Horizontal Edge
                if ((-22.5< Angle[i,j] <= 22.5) | ( -157.5 < Angle[i,j] <= 157.5)):
                    if((Magxy_Temp[i,j] < Magxy_Temp[i+1,j]) | (Magxy_Temp[i, j] < Magxy_Temp[i-1,j])):
                        NonMax[i,j] = 0
                        
                    
                #Vertical Edge
                if ((-112.5 < Angle[i,j] <= -67.5) | ( 67.5 < Angle[i,j] <= 112.5)):
                    if((Magxy_Temp[i,j] < Magxy_Temp[i,j+1]) | (Magxy_Temp[i, j] < Magxy_Temp[i,j-1])):
                        NonMax[i,j] = 0

                        
                #+45 Degree Edge
                
                if ((-67.5 < Angle[i,j] <= -22.5) | ( 112.5 < Angle[i,j] <= 157.5)):
                    if((Magxy_Temp[i,j] < Magxy_Temp[i+1,j+1]) | (Magxy_Temp[i, j] < Magxy_Temp[i+1,j+1])):
                        NonMax[i,j] = 0
                        

                #-45 degree Edge
                
                if ((-157.5 < Angle[i,j] <= -112.5) | (22.5 < Angle[i,j] <= 67.5 )):
                    if((Magxy_Temp[i,j] < Magxy_Temp[i-1,j-1]) | (Magxy_Temp[i, j] < Magxy_Temp[i+1,j+1])):
                        NonMax[i,j] = 0
                        
                

            except IndexError:
                    pass
                

    NonMax =(matrix(NonMax))
    
    # Apply hysteresis thresholding
    Hysterisis = deepcopy(NonMax)
    u = v = 0
    highT = 4.5 # The Non Maximum suppression matrix was checked for several points of 
    lowT = 1.5  #thresholds to suppress the non edge points
    maxm = 255 # we would be using this to set the pixel in order to make it a edge in the following for loops

    for i in range(len(Hysterisis[:,0])-1):
        
        for j in range(len(Hysterisis[0,:])-1):
            
            u = i
            v = j
            while((u!=0)&(v!=0)):
                
                if (Hysterisis[u,v] >=highT):
                    
                    Hysterisis[u,v] = maxm
                    try:
                        
                        if (lowT<=Hysterisis[u+1,v] < highT):
                            
                            Hysterisis[u+1,v] = maxm
                            u = u+1
                            v = v
                        elif (lowT<=Hysterisis[u-1,v]<highT):
                            
                            Hysterisis[u-1,v] = maxm
                            u = u-1
                            v= v
                        elif (lowT<=Hysterisis[u+1,v+1]<highT):
                            
                                    Hysterisis[u+1,y+1] = maxm
                                    u = u+1
                                    v = v+1
                        elif (lowT<=Hysterisis[u-1,v-1]<highT):
                                                    
                            Hysterisis[u-1,v-1] = maxm
                            u = u-1
                            v = v-1
                        elif (lowT<=Hysterisis[u,v+1]<highT):
                                                                           
                            Hysterisis[u,v+1] = maxm
                            u = u
                            v = v+1

                        elif (lowT<=Hysterisis[u,v-1]<highT):
                            
                            Hysterisis[u,v-1] = maxm
                            u = u
                            v = v-1
                        elif (lowT<=Hysterisis[u-1,v+1]<highT):
                            
                            Hysterisis[u-1,v+1] = maxm
                            u = u-1
                            v = v+1
                        elif (lowT<=Hysterisis[u+1,v-1]<highT):
                            
                            Hysterisis[u+1,v-1] = maxm
                            u = u+1
                            v = v-1
                        else: 
                            
                            u = 0
                            v = 0


                    except IndexError: 
                        
                        u = 0
                        v = 0

                elif (lowT<= Hysterisis[u,v]<highT):
                    
                    Hysterisis[u,v] = maxm

                else:
                    Hysterisis[u,v] = 0
                    u = 0
                    v = 0

    return I, Ixg, Iyg, Ix_gd, Iy_gd, Magxy, NonMax, Hysterisis

# Images and sigma values
images = ['image1.jpeg', 'image2.jpeg', 'image3.jpeg']
sigmas = [0.5, 1, 2]


# Create directory to save processed images
save_dir = "processed_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

# Process each image and save intermediate and final results
for idx, image in enumerate(images):
    image_name = os.path.splitext(os.path.basename(image))[0]
    I = np.array(Image.open(image).convert('L'))  # Convert image to grayscale before processing
    
    for i, sigma in enumerate(sigmas):
        results = canny_edge_detection(I, sigma)
        titles = ['Original', 'Ixg', 'Iyg', 'Ix_gd', 'Iy_gd', 'Magxy', 'NonMax', 'Hysterisis']

        # Save each processed image
        for j, (res, title) in enumerate(zip(results, titles)):
            save_name = os.path.join(save_dir, f"{image_name}_{title}_sigma_{sigma}.png")
            save_image(res, save_name)

for idx, image in enumerate(images):
    I = array(Image.open(image).convert('L'))  # Convert image to grayscale before processing

    fig, axs = plt.subplots(3, 8, figsize=(25, 20))
    fig.suptitle(f"Results for {image}")
    
    for i, sigma in enumerate(sigmas):
        results = canny_edge_detection(I, sigma)
        titles = ['Original Image', 'Image masked with Gaussian along x axis: Ix',
                  'Image masked with Gaussian along y axis: Iy', 
                  'Image masked with Gaussian Derivative along x axis : Ix`',
                  'Image masked with Gaussian Derivative along y axis : Iy`', 
                  'Image Edge Response Magnitude M(x,y)', 
                  'Non Maximum Suppressed Image', 'Hysterisiserisis Thresholding']
        
        for j, (res, title) in enumerate(zip(results, titles)):
            axs[i, j].imshow(res, cmap=cm.gray)
            axs[i, j].set_title(f"{title} (σ = {sigma})")
            axs[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

