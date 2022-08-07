# #!/usr/bin/env python  

#Reducer.py
import sys

dep_delay = {}

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
	
	if dep in dep_delay:
		dep_delay[dep].append(float(delay))
	else:
		dep_delay[dep] = []
		dep_delay[dep].append(float(delay))

#Reducer
for dep in dep_delay.keys():
	ave_dep = sum(dep_delay[dep])*1.0 / len(dep_delay[dep])
	min_dep = min(dep_delay[dep])
	max_dep = max(dep_delay[dep])
	print ('%s\t%s\t%s\t%s'% (dep, ave_dep,min_dep,max_dep))
