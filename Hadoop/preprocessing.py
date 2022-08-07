import numpy as np
import pandas as pd



df1 = pd.read_csv("D:/OneDrive/Desktop/jatin/ratings1.dat", names = ['movie_id','rating'],usecols=[1,2], dtype={'movie_id':np.float16,'rating':np.uint8},delimiter = '::',engine='python')
df2 = pd.read_csv("D:/OneDrive/Desktop/jatin/movies.dat", names = ['movie_id','genre'],dtype={'movie_id':np.float16}, usecols=[0,2],delimiter = '::',engine='python', encoding='latin-1')
df3=pd.merge(df1, df2, on='movie_id',how='right')[['genre','rating']]
df3.dropna(inplace=True)
path="D:\OneDrive\Desktop\Data Analytics\DDA LAB\Lab 6"

df3.to_csv('genre_rate1.csv', index=False)