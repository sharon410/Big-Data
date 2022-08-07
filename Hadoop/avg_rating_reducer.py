# #!/usr/bin/env python  

#Reducer.py
import sys
from operator import itemgetter


import pandas as pd

df = pd.read_csv('D:\OneDrive\Desktop\Data Analytics\DDA LAB\Lab 6\ml-10m\ml-10M100K\movies.dat',names = ['movie_id','title','genre'],delimiter = '::',engine='python')
movie_rating = {}
movie_list = {}
avg_rating=[]
#Partitoner
for line in sys.stdin:
	line = line.strip()
	split = line.split('\t')
	if len(split)>1:
		movie = split[0]
		rating = split[1]
	else:
		movie = split[0]
		rating = 0
	
	if movie in movie_list:
		movie_list[movie].append(float(rating))
	else:
		movie_list[movie] = []
		movie_list[movie].append(float(rating))


for movie in movie_list.keys():
	avg_rate= sum(movie_list[movie])*1.0 / len(movie_list[movie])
	avg_rating.append((movie,avg_rate))

final_list=[]
final_list=sorted(avg_rating, key=itemgetter(1),reverse=True)
max_value=max(final_list,key=itemgetter(1))
#Reducer
for movie_rating in final_list:
    if movie_rating[1]==max_value[1]:
        print(f"{df.loc[df.movie_id==int(movie_rating[0])]['title'].values[0]}\t{movie_rating[1]}")





