from mpi4py import MPI
import numpy as np
import scipy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_dataset():
    #This function is used for reading the dataset and returns tf-idf values for the words
    print("Dataset is read")
    newsgroups_train = fetch_20newsgroups(subset='train', remove = ('headers','footers','quotes'))
    vectorizer = TfidfVectorizer(stop_words = 'english')
    vectors = vectorizer.fit_transform(newsgroups_train.data)
    return vectors

def init_centroids(vec_tf_idf,k):
    #This function is used for initializing the centroids for the first time
    centroid_array=[]
    centroid_array=vec_tf_idf[np.random.choice(range(vec_tf_idf.shape[0]),k)]
    return centroid_array

def convert(a,b,n):

    res = np.zeros((1,n))
    res[0,b]=a
    return res

def computeDistance(arr,centroid_array):
    # This function returns the membership for each cluster
    cen_arr=centroid_array.todense()
    Ml=arr.tolil()
    Ml.data
    Ml.rows
    rowgen=(convert(a,b,101322) for a,b in zip(Ml.data,Ml.rows))
    dist_2=np.concatenate([scipy.spatial.distance.cdist(row,cen_arr, 'euclidean') for row in rowgen],axis=0)
    cluster = np.argmin(dist_2, axis=1)
    return cluster

def EuclideanDistance(vec,centroids):
    # This function returns the Euclidean distance between the data and the centroids
    cen_arr=centroids.todense()
    Ml=vec.tolil()
    Ml.data
    Ml.rows
    rowgen=(convert(a,b,101322) for a,b in zip(Ml.data,Ml.rows))
    dist_1=np.concatenate([scipy.spatial.distance.cdist(row,cen_arr, 'euclidean') for row in rowgen],axis=0)
    return dist_1.sum()

def computeCenter(vec_tf_idf,cen_arr):
    # This function is used to calculate the new cluster assignments and returns the new values of centroids after taking mean
    centroids_old=cen_arr.copy()
    clusters=[]
    cluster=computeDistance(vec_tf_idf,centroids_old) #Returns the membership array

    for i in range(centroids_old.shape[0]):  ##assign points to corresponding cluster
        clusters.append(vec_tf_idf[cluster==i]) 

    centroids_new=centroids_old.copy()
    for cluster_idx, cluster in enumerate(clusters):
        if cluster.shape[0]!=0:
            cluster_mean = np.mean(cluster, axis=0)
            centroids_new[cluster_idx] = cluster_mean
        else:
            centroids_new[cluster_idx]=centroids_old[cluster_idx]
    return centroids_new
        
if rank==0:
    ## Main Function.Four values of k are taken into consideration.k=5,7,9,15
    k=5
    # k=7
    # k=9
    # k=15
    start = MPI.Wtime()
    vec_tf_idf=read_dataset()
    global_centroids=init_centroids(vec_tf_idf,k) #Datapoints and k is passed as arguments
    splits = np.array_split(range(vec_tf_idf.shape[0]),size)
    vec_tf_idf_for_split = [vec_tf_idf[split] for split in splits]

else:
    vec_tf_idf_for_split=None
    global_centroids=None
    
vec_tf_idf_scatter=comm.scatter(vec_tf_idf_for_split,0)

flag_check_convergence = False

while not flag_check_convergence:

    global_centroids = comm.bcast(global_centroids,root=0)
    local_centroids= computeCenter(vec_tf_idf_scatter,global_centroids)
    old_centroids = global_centroids.copy() #A copy is saved for calculation of distance
    local_centroids = comm.gather(local_centroids.todense(),root=0)
    if rank==0:
        local_centroids = np.swapaxes(local_centroids,0,1)
        global_centroids = csr_matrix(local_centroids.mean(axis=1))
        dist1=EuclideanDistance(vec_tf_idf,old_centroids)
        dist2=EuclideanDistance(vec_tf_idf,global_centroids)
        dist_calculation_between_centroids=abs(dist1-dist2) #As it is distance calculation sign does not matter.
        print("Distance is",dist_calculation_between_centroids)
        if dist_calculation_between_centroids<5: #Check if newly assigned clusters are close to each other. Threshold value is taken into consideration
            # print("Inside Final Loop")
            flag_check_convergence=True
            print("New centroid assignment are",global_centroids)
            print("No of Workers are",size,"No of Clusters are ",k,"Time taken for execution of K-Means is",round(MPI.Wtime() - start,4))
    else:
        flag_check_convergence=False
    
    flag_check_convergence = comm.bcast(flag_check_convergence,root=0) #Final value of flag is broadcasted to all workers.








