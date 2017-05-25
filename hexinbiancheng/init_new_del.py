class C(object):

    count = 0

    @classmethod
    def add(cls):
        cls.count += 1

    @classmethod
    def howmany(cls):
        print cls.count

    def __init__(self, name):
        self.name = name
        self.add()
        print 'initialized'

    def __del__(self):
        # object.__delatr__(self)
        print self.name

c1 = C("c1")
c2 = c1
c3 = c1

print C.howmany()

print id(c1), id(c2), id(c3)
print getattr(c1, "name")
print dir(c1)
print c1.__dict__

del c1
del c2
del c3

print c2
