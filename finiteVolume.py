import numpy as np
import ray
import time

ray.init()

eta = 1
V = [eta,0]

T = 1

epsilon = 1e-9

normals = np.array([[0,-1],[-1,0],[0,1],[1,0]])

k = 4 
N = pow(2,k)
h = 1/N

cellsCount = pow(N,2)
    
def B1(z):
    return( 1 + max(-z,0) )

def u0(x,y):
    return( np.exp(eta*x) + np.exp(eta*x/2) * np.sin(np.pi*x) )

def uExact(x,y,t):
    return( np.exp(eta*x) + np.exp(-(np.pi*np.pi + eta*eta/4)*t + eta*x/2) * np.sin(np.pi*x) )


def solve_Finite_Volume():
    dt = h*h/(4*1.1)

    U = np.zeros((N,N),dtype="float")
    Uprime = np.zeros((N,N),dtype="float")

    for i in range(N):
        for j in range(N):
            U[i,j] = u0(j*h,i*h)
            
    t = 0

    while(t<T):
        t += dt

        for i in range(N):
            for j in range(N):
                s = 0

                for k in range(4):
                    VK_ij_sigma = normals[k,0]*V[0] + normals[k,1]*V[1]

                    if(j==0 and k==1):
                        s += 2*(B1(-VK_ij_sigma*h/2)*U[i,j] - B1(VK_ij_sigma*h/2))
                    elif(j==(N-1) and k==3):
                        s += 2*(B1(-VK_ij_sigma*h/2)*U[i,j] - B1(VK_ij_sigma*h/2)*np.exp(1))

                    # Neumann EGES
                    elif(i==0 and k==2):
                        pass
                    elif(i==(N-1) and k==0):
                        pass
                    
                    # Interior EDGES
                    else:
                        i_neighbour = i - normals[k][1]
                        j_neighbour = j + normals[k][0]
                        
                        s += B1(-VK_ij_sigma*h)*U[i,j] - B1(VK_ij_sigma*h)*U[i_neighbour,j_neighbour]

                Uprime[i,j] = U[i,j] - dt*s/(h*h)
 
        for i in range(N):
            for j in range(N):
                U[i,j] = Uprime[i,j]
		    
    # print("Final t = " + str(t))
    
    # To compare the vector U with the exact solution #
    error = 0;
    for i in range(N):
        for j in range(N):
            error += h*h*pow(U[i,j]-uExact( j*h , i*h , t) , 2);
            
    error = pow(error,.5);

    # print("")
    # print("Error with the exact solution at time T = " + str(error) )
    # print("")

@ray.remote
def solve_Finite_Volume_ray():
    dt = h*h/(4*1.1)

    U = np.zeros((N,N),dtype="float")
    Uprime = np.zeros((N,N),dtype="float")

    for i in range(N):
        for j in range(N):
            U[i,j] = u0(j*h,i*h)
            
    t = 0

    while(t<T):
        t += dt

        for i in range(N):
            for j in range(N):
                s = 0

                for k in range(4):
                    VK_ij_sigma = normals[k,0]*V[0] + normals[k,1]*V[1]

                    if(j==0 and k==1):
                        s += 2*(B1(-VK_ij_sigma*h/2)*U[i,j] - B1(VK_ij_sigma*h/2))
                    elif(j==(N-1) and k==3):
                        s += 2*(B1(-VK_ij_sigma*h/2)*U[i,j] - B1(VK_ij_sigma*h/2)*np.exp(1))

                    # Neumann EGES
                    elif(i==0 and k==2):
                        pass
                    elif(i==(N-1) and k==0):
                        pass
                    
                    # Interior EDGES
                    else:
                        i_neighbour = i - normals[k][1]
                        j_neighbour = j + normals[k][0]
                        
                        s += B1(-VK_ij_sigma*h)*U[i,j] - B1(VK_ij_sigma*h)*U[i_neighbour,j_neighbour]

                Uprime[i,j] = U[i,j] - dt*s/(h*h)
 
        for i in range(N):
            for j in range(N):
                U[i,j] = Uprime[i,j]
		    
    # print("Final t = " + str(t))
    
    # To compare the vector U with the exact solution #
    error = 0;
    for i in range(N):
        for j in range(N):
            error += h*h*pow(U[i,j]-uExact( j*h , i*h , t) , 2);
            
    error = pow(error,.5);

    return(U)
    # print("")
    # print("Error with the exact solution at time T = " + str(error) )
    # print("")

print("Naive Python + Ray Implementation :")
tStart = time.time()
for i in range(5):
    U = solve_Finite_Volume_ray.remote()
    print(ray.get(U))
tEnd = time.time()
print(tEnd-tStart)

print("Naive Python Implementation :")
tStart = time.time()
for i in range(5):
    solve_Finite_Volume()
tEnd = time.time()
print(tEnd-tStart)



ray.shutdown()
