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

    v1=np.random.rand(10000)
    
    print("Vector is",v1)
    
    # master process sends data to worker processes by
    # going through the ranks of all worker processes
    data1=np.array_split(v1, size-1)
    # print("After split is",data1)
    
    for i in range(size-1):
        comm.send(data1[i], dest=i+1)
       
    for i in range(size-1):
        sum2=0
        data_received=comm.recv(source=i+1)
        
        arr_sum.append(data_received)
        for i in range(len(arr_sum)):
            sum2+=arr_sum[i]
        sum +=(time.time()- start_time)
    print("Total time taken",sum)
    print("Average value of the vector is",(sum2/len(arr_sum)))
    
    
# worker processes
else:
    # each worker process receives data from master process
    data = comm.recv(source=0)
    sum1=0
    for i in range(len(data)):
        sum1+=data[i]

    avg=(sum1/len(data))
    # print("Avg inside Worker",avg)
    comm.send(avg,dest=0)
    # print('Process {} received data:'.format(rank), data)

