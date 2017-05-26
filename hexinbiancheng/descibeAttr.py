class DevNull(object):

    def __get__(self, obj, type=None):
        pass

    def __set__(self, obj, val):
        pass


class C1(object):
    foo = DevNull()

c1 = C1()
c1.foo = 'bar'

print 'c1.foo contains:', c1.foo


class DevNull2(object):

    def __get__(self, obj, type=None):
        print 'Accessing attribute ... ignoring'

    def __set__(self, obj, val):
        print 'Attempt to assign %r ... ignoring' % val


class C2(object):
    foo = DevNull2()

c2 = C2()
c2.foo = 'bar'


class DevNull3(object):

    def __init__(self, name=None):
        self.name = name

    def __get__(self, obj, type=None):
        print 'Accessing [%s] ... ignoring' % self.name

    def __set__(self, obj, val):
        print 'Assigning %r to [%s] ... ignoring' % (val, self.name)


class C3(object):
    foo = DevNull3('foo')

c3 = C3()
c3.foo = 'bar'
