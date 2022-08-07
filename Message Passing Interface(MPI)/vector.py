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

    v1=np.random.rand(10)
    v2=np.random.rand(10)
    print("\n First vector is",v1)
    print("\n Second vector is",v2)
    # master process sends data to worker processes by
    # going through the ranks of all worker processes
    
    data1=np.array_split(v1, size-1)
    data2=np.array_split(v2, size-1)
   
    for i in range(size-1):
        comm.send([data1[i],data2[i]], dest=i+1)

    for i in range(size-1):
        data_received=comm.recv(source=i+1) 
        arr_sum.append(data_received)
        sum +=(time.time()- start_time)
    print("Addition of two vectors is",np.concatenate(arr_sum).ravel())
    print("Total time taken",sum)
   
    
# worker processes
else:
    # each worker process receives data from master process
    data = comm.recv(source=0)
    sum1=data[0]+data[1]
    # print("Sum inside Worker",sum1)
    comm.send(sum1,dest=0)
    # print('Process {} received data:'.format(rank), data)

