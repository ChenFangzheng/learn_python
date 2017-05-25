#!/usr/bin/env python
# -*-coding:utf-8-*-
from time import ctime, sleep, time


def tsfunc(func):
    '''装饰器的返回值是一个"包装了"的函数'''
    def wrappedFunc():
        print '[%s] %s() called' % (ctime(), func.__name__)
        return func
    return wrappedFunc


@tsfunc
def foo():
    pass

foo()
sleep(4)

for i in range(2):
    sleep(1)
    foo()

'''-----------------advance example ------------------'''


def logged(when):
    def log(f, *args, **kargs):
        print ''' Called:
        function: %s
        args: %r
        kargs: %r''' % (f, args, kargs)

    def pre_logged(f):
        def wrapper(*args, **kargs):
            log(f, *args, **kargs)
            return f(*args, **kargs)
        return wrapper

    def post_logged(f):
        def wrapper(*args, **kargs):
            now = time()
            try:
                return f(*args, **kargs)
            finally:
                log(f, *args, **kargs)
                print "time delta: %s" % (time() - now)
        return wrapper

    try:
        return {"pre": pre_logged,
                "post": post_logged}[when]
    except KeyError, e:
        raise ValueError(e), 'must be "pre" or "post"'


@logged('post')
def hello(name):
    print "Hello,", name

hello("world!")
