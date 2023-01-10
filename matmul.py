import numpy as np
import ray
import time

ray.init()

N = 100

def multiply(A,B):
    C = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            s = 0
            for k in range(N):
                s += A[i,k] * B[k,j]
            C[i,j] = s
    
    return(C.copy())

#Same code, with ray !
@ray.remote
def multiply_ray(A,B):
    C = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            s = 0
            for k in range(N):
                s += A[i,k] * B[k,j]
            C[i,j] = s
    
    return(C.copy())

A = np.ones((N,N))
B = np.ones((N,N))

print("Very naive :")
tStart = time.time()
for i in range(10):
	C = multiply(A,B)
	print(C)
tEnd = time.time()
print(tEnd-tStart)

print("Numpy multiplication :")
tStart = time.time()
for i in range(10):
	C = np.matmul(A,B)
	print(C)
tEnd = time.time()
print(tEnd-tStart)

print("Ray multiplication :")
tStart = time.time()
for i in range(10):
	C = multiply_ray.remote(A,B)
	print(ray.get(C))
tEnd = time.time()
print(tEnd-tStart)


ray.shutdown()








