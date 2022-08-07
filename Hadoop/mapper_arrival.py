#!/usr/bin/env python
  
# import sys because we need to read and write data to STDIN and STDOUT
import sys
flag=False
# reading entire line from STDIN (standard input)
for line in sys.stdin:
    # to remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    tokens = line.split(',') 
      
    
    if flag:
        airport  = tokens[4]  
        if (tokens[8]!=''):
            arrival_delay = int(float(tokens[6]))
            # print(dep_delay)
            print ('%s\t%s' % (airport, arrival_delay))
    flag=True
    