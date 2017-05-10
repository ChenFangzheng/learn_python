aString = 'abcd'
print aString[1:3]
print aString[3]


print aString.index('bc')
print aString.find('c')


lis = ['1', '100', '111', '2']
mNum = max(lis, key=lambda x: int(x))
print mNum

lis2 = [(1, 'a'), (3, 'c'), (4, 'e'), (-1, 'z')]
mNum2 = max(lis2, key=lambda x: x[0])
print mNum2


lis3 = [1, 2, 4, 5, 6]
lis4 = [2, 4, 5, 6, 6]
print zip(lis3, lis4)


'''stra = "we are"
strb = "both boy!"
print ' '.join((stra, strb))

li1 = ['11', '22']
li2 = ['33', '44']

li1.extend(li2)
print li1


s = "I know you are a good boy!"

for i in [None] + range(-1, -len(s), -1):
    print s[:i]
'''
