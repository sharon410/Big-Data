# #!/usr/bin/env python  

#Reducer.py
import sys
from operator import itemgetter

arrival_delay = {}
avg_delay=[]
#Partitoner
for line in sys.stdin:
	line = line.strip()
	split = line.split('\t')
	if len(split)>1:
		dep = split[0]
		delay = split[1]
	else:
		dep = split[0]
		delay = 0
	
	if dep in arrival_delay:
		arrival_delay[dep].append(float(delay))
	else:
		arrival_delay[dep] = []
		arrival_delay[dep].append(float(delay))


for dep in arrival_delay.keys():
	ave_dep = sum(arrival_delay[dep])*1.0 / len(arrival_delay[dep])
	avg_delay.append((dep,ave_dep))

final_list=[]
final_list=sorted(avg_delay, key=itemgetter(1),reverse=True)
#Reducer
for rank,delay in enumerate(final_list):
    if rank<10:
        print ('%s\t%s\t%s'% (rank+1,delay[0],delay[1]))





