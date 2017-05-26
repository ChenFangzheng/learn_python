class HideX(object):

    @property
    def x(self):
        return self.__x

    @x.getter
    def x(self):
        return ~self.__x

    @x.setter
    def x(self, x):
        assert isinstance(x, int), '"x" must be an interger!'
        self.__x = ~x

inst = HideX()

inst.x = 50

print inst.x

inst.x = 30

print inst.x
