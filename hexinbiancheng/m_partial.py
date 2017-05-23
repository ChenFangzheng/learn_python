from operator import add, mul
from functools import partial

add1 = partial(add, 1)
mul100 = partial(mul, 100)

print add1(10)
print mul100(10)

baseTwo = partial(int, base=2)

print baseTwo('10010')
