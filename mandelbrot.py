import numpy as np
import matplotlib.pyplot as plt
import ray
import time

ray.init()

x_min = -1.78
x_max = 0.78
y_min = -0.961
y_max = 0.961


def Compute (im , width , height , nb_iter, x_min, x_max, y_min, y_max):
      
    pos = 0
          
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height
      
    for l in range(height):
        for c in range(width):
            a = x_min + c * dx
            b = y_max - l * dy
            x = 0
            y = 0
            i=0
            while (i < nb_iter):
                tmp = x
                x = x * x - y * y + a
                y = 2 * tmp * y + b
                if (x * x + y * y > 4):
                    break
                else:
                    i+=1

            im[pos//width][pos%width] =  i / nb_iter * 255
            pos += 1

@ray.remote
def Compute_ray (im , width , height , nb_iter, x_min, x_max, y_min, y_max):

    im2 = im.copy()
    
    pos = 0
          
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height
      
    for l in range(height):
        for c in range(width):
            a = x_min + c * dx
            b = y_max - l * dy
            x = 0
            y = 0
            i=0
            while (i < nb_iter):
                tmp = x
                x = x * x - y * y + a
                y = 2 * tmp * y + b
                if (x * x + y * y > 4):
                    break
                else:
                    i+=1

            im2[pos//width][pos%width] =  i / nb_iter * 255
            pos += 1

    return(im2.copy())
  
nb_iter = 20
width = 512
height = 420

tStart = time.time()
for i in range(5):
    im = np.zeros((height,width),dtype='int')
    Compute (im, width, height , nb_iter, x_min, x_max, y_min, y_max)

    if(i==0):
        plt.imsave("Serial.png",im/nb_iter)
        
tEnd = time.time()
print("5 serial executions : " + str(tEnd - tStart) )


tStart = time.time()
for i in range(5):
    im = np.zeros((height,width),dtype='int')
    im = Compute_ray.remote(im, width, height , nb_iter, x_min, x_max, y_min, y_max)
    im = ray.get(im)
    if(i==0):
        plt.imsave("Ray.png",im/nb_iter)
        
tEnd = time.time()
print("5 parallel executions : " + str(tEnd - tStart) )



ray.shutdown()
