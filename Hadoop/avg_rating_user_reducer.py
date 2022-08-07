# #!/usr/bin/env python  

#Reducer.py
import sys
from operator import itemgetter

user_list = {}
user_list1 = {}
avg_rating={}
#Partitoner
for line in sys.stdin:
	line = line.strip()
	split = line.split('\t')
	if len(split)>1:
		user = split[0]
		rating = split[1]
	else:
		user = split[0]
		rating = 0
	
	if user in user_list:
		user_list[user].append(float(rating))
	else:
		user_list[user] = []
		user_list[user].append(float(rating))


for user in user_list.keys():
    if len(user_list[user])>40:
        user_list1[user]=user_list[user]

for user in user_list1.keys():
	avg_rating[user]= sum(user_list1[user])*1.0 / len(user_list1[user])

min_value=min(avg_rating.values())
#Reducer
for user in user_list1.keys():
    if avg_rating[user]==min_value:
        print ('%s\t%s'% (user,avg_rating[user]))





