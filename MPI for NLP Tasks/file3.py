from mpi4py import MPI
import time
import numpy as np
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print_time = False

arr_sum=[]
arr_tf=[]
idf_main_array={}
tf_idf_main_array=[]

### Read Data from Location
def read_data(path):
    if print_time: start = MPI.Wtime()
    file_line=[]
    for file in os.listdir(path):
        filepath=f"{path}/{file}"
        for file1 in os.listdir(filepath):

            filepath1=f"{filepath}/{file1}"
                # print(filepath1)
            with open(filepath1,"r") as f:
                lines = [line.strip() for line in f]
                file_line.append(lines)

    if print_time: print("Rank: ",rank,"Reading Data: ",
                         round(MPI.Wtime() - start,4))
    return file_line

#### Perform Tokenisation and removal of Stop Words
def clean_data(data):
    stop_words = set(stopwords.words("English"))
    filtered_sentence=[]
    main_line=[]
    ar3=[]
    ar4=[]
    len1=len(data)
    token_arr=[]
    ###Tokenisation
    for j in range(len1):
        for i in range(len(data[j])):
            text_only= re.sub("[^a-zA-Z]"," ",data[j][i])
            text = text_only.lower()
            token_arr.append(word_tokenize(text))
        flat_list = [item for sublist in token_arr for item in sublist]
        main_line.append(flat_list)
        
    #Removal of Stop Words
    for i in range(len(main_line)):
        filtered_sentence=[]
        for w in main_line[i]:
            if w not in stop_words:
                filtered_sentence.append(w)
        ar3.append(filtered_sentence)
 
    return ar3

### Count Term Frequency- Here results and stored in a Dictionary and key is the word
def count_tf(rec):
    if print_time: start = MPI.Wtime()
    list3=[]
    for i in range(len(rec)):
        dict1={}
        dict2={}
        word_counts = Counter(rec[i])
        dict1=dict(word_counts)
        for key1,value1 in dict1.items():
            # print(key1,value1)
            key=key1
            dict2[key]=(value1/len(rec[i]))
        list3.append(dict2)
    if print_time: print("Rank: ",rank,"Counting TF Tokens from Data: ",
                         round(MPI.Wtime() - start,4))
    return list3


def find_tokens_in_data(arr_rec):
    token_arr=[]
    token_arr_new=[]
    for i in range(len(arr_rec)):
        for j in range(len(arr_rec[i])):
            for k in range(len(arr_rec[i][j])):
                token_arr.append(arr_rec[i][j][k])
      
    myset=set(token_arr)
    token_arr_new=list(myset)
    return token_arr_new

### Calculate IDF Value 
def calculate_idf(arr_rec,token_split):
    if print_time: start = MPI.Wtime()
    count_idf={}
    inter_dict={}
    
    for token in token_split:
        inter_dict[token]=1
        
    for i in range(len(arr_rec)):
        # print("Inside")
        for j in token_split:
            check_in_doc=list(set(arr_rec[i]))
            if j in check_in_doc:
                inter_dict[j]+=1           
    
    for key, value in inter_dict.items():
        count_idf[key]=np.log(len(data_1)/value)

    if print_time: print("Rank: ",rank,"Calculating IDF Data: ",
                         round(MPI.Wtime() - start,4))   
    return count_idf

#### Calculate TF-IDF Value
def calculate_tf_idf(list3,count_idf):
    if print_time: start = MPI.Wtime()
    dict_tf_idf={}
    for i in range(len(list3)):
        for key1,value1 in count_idf.items():
            key=key1
            if key1 in list3[i].keys():
                dict_tf_idf[key]=list3[i][key1]*count_idf[key1] 
    

    if print_time: print("Rank: ",rank,"TF-IDF Value : ",
                         round(MPI.Wtime() - start,4))
    return dict_tf_idf
    
                            
if rank==0:
    i=0
    data_rec=[]
    path="D:/OneDrive/Desktop/Data Analytics/DDA LAB/Lab 3/newsgroups"

    data_1=read_data(path)

    # print("Length of Data is",len(data_1))
    arr1=np.array_split(data_1,size-1)
    start = MPI.Wtime()
    for i in range(size-1):
        # print("Send from master")

        comm.send(arr1[i], dest=i+1,tag=1)
        # comm.send(["Hello"], dest=i+1)
        # print("Received to master")
        data_received=comm.recv(source=i+1,tag=2)
        # print(len(data_received))

        comm.send(data_received,dest=i+1,tag=3)
        tf_data_list=comm.recv(source=i+1,tag=4)
        arr_sum.append(data_received)
        arr_tf.append(tf_data_list)
    total_tokens=find_tokens_in_data(arr_sum)
    token_split=np.array_split(total_tokens,size-1)
    # print("Final TF Calculation dict is",arr_tf[0][0])
    # print("Final Data Value After Cleaning is",arr_sum[0][0])
    # print("No of Workers are ",size,"Time taken for execution is",round(MPI.Wtime() - start,4))

    ### IDF Calculation
    for i in range(size-1):
        # print("Iteration",i)
        comm.send([arr_sum[i],token_split[i]],dest=i+1,tag=5)
        idf_value=comm.recv(source=i+1,tag=6)
        idf_main_array.update(idf_value)
    # print("Final IDF Token Array is",idf_main_array)
    # print("No of Workers are ",size,"Time taken for execution of IDF is",
    #                      round(MPI.Wtime() - start,4))

    ##### TF- IDF Calculation
    for i in range(size-1):
        # print("Iteration",i)
        comm.send([arr_tf[i],idf_main_array],dest=i+1,tag=7)
        tf_idf_value=comm.recv(source=i+1,tag=8)
        tf_idf_main_array.append(tf_idf_value)
    print("Final IDF Token Array is",tf_idf_main_array[0])
    print("No of Workers are ",size,"Time taken for execution of TF-IDF is",
                         round(MPI.Wtime() - start,4))

else:
    # each worker process receives data from master process
    data = comm.recv(source=0,tag=1)
    comm.send(clean_data(data),dest=0,tag=2)
    tf_data=comm.recv(source=0,tag=3)
    comm.send(count_tf(tf_data),dest=0,tag=4)
    token_data=comm.recv(source=0,tag=5)
    comm.send(calculate_idf(token_data[0],token_data[1]),dest=0,tag=6)
    tf_idf_calculation=comm.recv(source=0,tag=7)
    comm.send(calculate_tf_idf(tf_idf_calculation[0],tf_idf_calculation[1]),dest=0,tag=8)