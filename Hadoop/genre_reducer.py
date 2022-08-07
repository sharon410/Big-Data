# #!/usr/bin/env python  

#Reducer.py
import sys


genre_list = {}
avg_rating={}
#Partitoner
for line in sys.stdin:
	line = line.strip()
	split = line.split('\t')
	if len(split)>1:
		genre = split[0]
		rating = split[1]
	else:
		genre = split[0]
		rating = 0
	
	if genre in genre_list:
		genre_list[genre].append(float(rating))
	else:
		genre_list[genre] = []
		genre_list[genre].append(float(rating))


for genre in genre_list.keys():
	avg_rating[genre]= sum(genre_list[genre])*1.0 / len(genre_list[genre])

max_value=max(avg_rating.values())

#Reducer
for genre in genre_list.keys():
    if avg_rating[genre]==max_value:
        print ('%s\t%s'% (genre,max_value))





