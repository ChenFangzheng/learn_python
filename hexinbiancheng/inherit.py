class P(object):

    def __init__(self):
        print 'calling P init'

    def foo(self):
        print 'I am P-foo'


class C(P):

    def __init__(self):
        super(C, self).__init__()
        # P.__init__(self)
        print 'calling C init'

    def foo(self):
        super(C, self).foo()
        print 'Hi, I am C-foo'


c = C()
print c.foo()


class RoundFloat(float):

    def __new__(cls, val):
        return super(RoundFloat, cls).__new__(cls, round(val, 2))


print RoundFloat(1.56645)


class RoundFloatManual(object):

    def __init__(self, value):
        assert isinstance(value, float), "Value must be a float"
        self.value = round(value, 2)

    def __str__(self):
        return '%.2f' % self.value

    __repr__ = __str__

rfm = RoundFloatManual(34.0000)

print rfm
