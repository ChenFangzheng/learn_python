# -*-coding:utf-8-*-

import copy

person = ['name', ['savings', 100]]
hubby = person
wifey = copy.deepcopy(person)
print [id(x) for x in person, hubby, wifey]

print [id(x) for x in hubby]
print [id(x) for x in wifey]

hubby[0] = 'joe'
wifey[0] = 'jane'

print hubby, wifey

hubby[1][1] = 500.00
print hubby, wifey

print [id(x) for x in hubby]
print [id(x) for x in wifey]

print '---------------------'
# 非容器类型美欧拷贝一说, 如果tuple变量只包含原子类型队形,对它进行的身拷贝不会进行
tPerson = ['name', ('savings', 100)]
tNewPerson = copy.deepcopy(tPerson)
print [id(x) for x in tPerson, tNewPerson]
print [id(x) for x in tPerson]
print [id(x) for x in tNewPerson]
tPerson[0] = 'test'
print 'after change'
print [id(x) for x in tPerson]
print [id(x) for x in tNewPerson]
