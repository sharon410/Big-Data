from mpi4py import MPI
import numpy as np
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_data_kdd():
    #Read the Data
    #Perform Label  encoding
    #Normalize the Data
    #Split the data into train and test based on 70%:30% split
    #Return test and train
    kdd_df=pd.read_csv("D:\\OneDrive\\Desktop\\Data Analytics\\DDA LAB\\Lab 5\\cup98LRN.txt",delimiter=',',low_memory=False)
    #Label Encoding
    for col in kdd_df.columns:
        if kdd_df[col].dtype=='object':
            kdd_df[col] = kdd_df[col].astype('category')
            kdd_df[col] = kdd_df[col].cat.codes

    normalized_df=(kdd_df-kdd_df.mean())/kdd_df.std()
    df1=normalized_df.fillna(0)
    kdd_df_train=df1.sample(frac=0.7,random_state=200) #random state is a seed value
    kdd_df_test=df1.drop(kdd_df_train.index)
    return kdd_df_train,kdd_df_test


def find_least_square(kdd_df,global_beta):

    X=kdd_df.drop(columns=['TARGET_D','TARGET_B','CONTROLN','RFA_2R'],inplace=False)
    y=kdd_df["TARGET_D"]
    m=len(y)
    X = np.hstack((np.ones([m,1]), X)) # Append the bias term (field containing all ones) to X.
    y = np.array(y).reshape(-1,1) 

    yhat = np.dot(X,global_beta)
    costtrain = np.sqrt(np.mean(pow((y-yhat),2)))
    return costtrain


def stochasticgradientdescent(kdd_df_train,beta):
    #Define X and y
    
    X=kdd_df_train.drop(columns=['TARGET_D','TARGET_B','CONTROLN','RFA_2R'],inplace=False)
    y=kdd_df_train["TARGET_D"]
    m=len(y)
    X = np.hstack((np.ones([m,1]), X)) # Append the bias term (field containing all ones) to X.
    y = np.array(y).reshape(-1,1) 
    learning_rate = 1e-4
    index = np.arange(0,len(X)) 
    np.random.shuffle(index)
    for i in index:
        X_ele=X[i].reshape(-1,1)

        temp=np.dot(X_ele.T,beta)-y[i]
        temp=np.dot(X_ele,temp)
        gradient = temp * (2)
        beta = beta - (learning_rate * gradient)
    return beta

if rank==0:
    #After test and train is returned, split the train based on the number of workers 
    start = MPI.Wtime()
    kdd_df_train,kdd_df_test=read_data_kdd() ##Reads data into a dataframe
    kdd_df_train=kdd_df_train.iloc[0:1000,:]
    splits = np.array_split(range(kdd_df_train.shape[0]),size)
    kdd_df_train_split = [kdd_df_train.iloc[split,:] for split in splits]

else:
    kdd_df_train_split=None


kdd_df_train_split=comm.scatter(kdd_df_train_split,0)
beta=np.zeros([kdd_df_train_split.shape[1]-3,1])
flag_check_convergence = False
cost_train = []
cost_test = []

while not flag_check_convergence:

    local_beta= stochasticgradientdescent(kdd_df_train_split,beta)
    old_beta=beta.copy()

    global_beta=comm.allreduce(local_beta,op=MPI.SUM)
    beta = global_beta / size
    

    if rank==0:
        #Make two array for train rmse and test rmse
        costtrain=find_least_square(kdd_df_train,beta)
        cost_train.append(costtrain)
        
        costtest=find_least_square(kdd_df_test,beta)
        cost_test.append(costtest)
        diff1=abs(find_least_square(kdd_df_train,old_beta)-find_least_square(kdd_df_train,beta))
        if diff1<0.000001:  #Checking for Convergence criteria
            print("Inside Final Loop")
            flag_check_convergence=True
            path="D:\\OneDrive\\Desktop\\"
            df = pd.DataFrame({'TrainRMSE':cost_train,'TestRMSE': cost_test,'Time' :round(MPI.Wtime() - start,4)})
            df.to_csv(path+'KDDCup'+str(size)+'.csv')
            print("No of Workers are",size,"Time taken for execution is",round(MPI.Wtime() - start,4))
    else:
        flag_check_convergence=False
    
    flag_check_convergence = comm.bcast(flag_check_convergence,root=0) #Final value of flag is broadcasted to all workers.