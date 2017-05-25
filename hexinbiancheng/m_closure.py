#!/usr/bin/env python
#-*-coding:utf-8-*-
'''
    如果在一个内部函数里,对在外部作用域(但不是全局作用域)的变量进行引用,那么内部函数就被认为是closure
    定义在外部函数内的但由内部函数引用或者使用的变量被称为自由变量
'''


def counter(start_at=0):
    count = [start_at]

    def incr():
        count[0] += 1
        return count[0]
    return incr


count = counter(5)
print count()

count2 = counter(100)
print count2()

print count()
