#!/usr/bin/env ptyhon


def convert(func, seq):
    '''conv. sqequence of numbers to the same type'''
    return [func(num) for num in seq]

myseq = (123, 45.67, -6.2e8, 999999999L)

print convert(int, myseq)
print convert(float, myseq)
print convert(long, myseq)
