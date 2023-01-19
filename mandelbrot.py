import numpy as np
import matplotlib.pyplot as plt
import ray
import time

ray.init(num_cpus = 2)

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
            im2[pos//width][pos%width] = i/nb_iter*255
            pos += 1
    return im2.copy()

nb_iter = 500
width = 2*512
height = 2*420

#tStart = time.time()
#im = np.zeros((height,width),dtype='int')
#Compute(im, width, height , nb_iter, x_min, x_max, y_min, y_max)
#tEnd = time.time()
#plt.imsave("Serial.png", im/nb_iter)
#print("Serial execution : " + str(tEnd - tStart))

tStart = time.time()
im = np.zeros((height,width),dtype='int')
results = ray.get([Compute_ray.remote(im[:420,:], width, 420, nb_iter, x_min, x_max, 0, y_max),Compute_ray.remote(im[420:,:], width, 420, nb_iter, x_min, x_max, y_min, 0)])
im[:420,:] = results[0]
im[420:,:] = results[1]
tEnd = time.time()
plt.imsave("Ray.png", im/nb_iter)
print("Parallel execution : " + str(tEnd - tStart))

ray.shutdown()
