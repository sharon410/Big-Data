#!/usr/bin/env python
  
# import sys because we need to read and write data to STDIN and STDOUT
import sys
flag=False
# reading entire line from STDIN (standard input)
for line in sys.stdin:
    # to remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    genre,rating = line.split(',') 
      
    if flag:
        print ('%s\t%s' % (genre,rating))

    flag=True
    