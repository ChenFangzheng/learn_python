#!/usr/bin/env python

from time import ctime

print '*** welcome to metaclasses!'


print '\t Metaclass declaratin frist.'


class MetaC(type):

    def __init__(cls, name, bases, attrd):
        super(MetaC, cls).__init__(name, bases, attrd)
        print '** Created class %r at: %s' % (name, ctime())
        print attrd

print '\t Class "foo" declaration next.'


class Foo(object):
    __metaclass__ = MetaC

    def __init__(self):
        print '** Instantiated class %r at %s' % (self.__class__.__name__, ctime())

print '\t Class "Foo" instantiation next.'

f = Foo()

print '\t DONE'
