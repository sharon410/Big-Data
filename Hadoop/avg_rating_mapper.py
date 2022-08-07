#!/usr/bin/env python
  
# import sys because we need to read and write data to STDIN and STDOUT
import sys

# reading entire line from STDIN (standard input)
for line in sys.stdin:
    # to remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    user_id,movie_id,rating,timestamp = line.split('::') 
       
    print ('%s\t%s' % (movie_id, rating))
    
