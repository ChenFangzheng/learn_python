#!/usr/bin/env python

'''create text file'''

import os
ls = os.linesep

# get file name

while True:
    fname = raw_input("Enter your filename: ")
    if os.path.exists(fname):
        print "Error: '%s' already exists" % fname
    else:
        break

# get file conent (text) lines
all = []
print "\nEnter lines ('.' by itself to quit).\n"

# loop until user terminates input

while True:
    entry = raw_input('>')
    if entry == '.':
        break
    else:
        all.append(entry)

# write lines to file with proper line-ending
fobj = open(fname, 'w')
fobj.writelines(['%s%s' % (x, ls) for x in all])
fobj.close()
print 'DONE!'
