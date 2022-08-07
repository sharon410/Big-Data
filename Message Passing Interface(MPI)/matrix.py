from mpi4py import MPI
import time
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
N=1000

master=0
sum=0
if rank==0:
    start_time = time.time()
    mat1 = np.random.rand(N,N)
    mat2 = np.random.rand(N,N)
    mat3 = np.random.rand(N,N)
    mat1_split=np.array_split(mat1,size)
    # print(mat1_split)

    print("Matrix 1 is",mat1)
    print("Matrix 2 is",mat2)
else:
    # mat1 = np.random.rand(N,N)
    # mat2 = np.random.rand(N,N)
    mat1 = np.zeros((N,N))
    mat2 = np.zeros((N,N))
    mat1_split=np.array_split(mat1,size)
    # print(mat1_split)

comm.Bcast(mat2,master)
for i in range(len(mat1_split)):

    data=comm.scatter(mat1_split,master)
    # print("Data after scatter is",data)

    mat3=comm.gather(np.dot(data,mat2),master)

if rank==0:
    
    print("Matrix Multiplication is",mat3)
    sum +=(time.time()- start_time)
    print("Total time taken for multiplication is",sum)