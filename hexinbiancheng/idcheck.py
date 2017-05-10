#!/usr/bin/env python

import string
from string import Template


s, t, a = 'abcde', '12345', 'ABCDE'
print zip(s, t, a)

alphas = string.letters + '_'
nums = string.digits

s = Template('there are ${num} peers here!')
print s.substitute(num=5)

print 'Welcom to the Identifier checker v1.0'
print 'testees must be at least 2 chars long.'

inp = raw_input('Identifier to test?')

if len(inp) > 1:
    if inp[0] not in alphas:
        print ''' invalid: first symbol must be
            alphabetic'''
    else:
        alpnums = alphas + nums
        for otherChar in inp[1:]:
            if otherChar not in alpnums:
                print ''' Invalid: remaining symbol must be alphanumeric'''
                break
        else:
            print "okay as an identifier"
