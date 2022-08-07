from mpi4py import MPI
import time
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#master process
sum=0
arr_sum=[]
if rank == 0:
    start_time = time.time()
    print("Start Time is",start_time)

    v1=np.random.rand(10,10)
    v2=np.random.rand(10)
    print("Matrix 1 is",v1)
    print("Vector is",v2)
    # master process sends data to worker processes by
    # going through the ranks of all worker processes
    data1=np.array_split(v1, size-1)
    # # print("After split is",data1)
    # data2=np.array_split(v2, size-1)
    # print("After split is",data2)
    for i in range(size-1):
        comm.send([data1[i],v2], dest=i+1)

       
    for i in range(size-1):
        data_received=comm.recv(source=i+1)
        
        arr_sum.append(data_received)
        
        sum +=(time.time()- start_time)
    print("Total time taken",sum)
    print("Multiplication is",arr_sum)
    
    
# worker processes
else:
    # each worker process receives data from master process
    data = comm.recv(source=0)
    sum1=np.empty([len(data[0]),1])
    for i in range(len(data[0])):
        for j in range(len(data[1])):
            sum1[i]+=data[0][i][j]*data[1][j]
    # sum1=np.matmul(data[0],data[1])
    # print("Sum inside Worker",sum1)
    comm.send(sum1,dest=0)
    # print('Process {} received data:'.format(rank), data)

