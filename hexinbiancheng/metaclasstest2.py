from warnings import warn


class ReqStrSugRepr(type):

    def __init__(cls, name, bases, attrd):

        super(ReqStrSugRepr, cls).__init__(name, bases, attrd)

        if '__str__' not in attrd:
            raise TypeError("Class requires overriding of __str__()")
        if '__repr__' not in attrd:
            warn('Class suggests overriding of __repr__()\n', stacklevel=3)


class Foo(object):

    __metaclass__ = ReqStrSugRepr

    def __str__(self):
        return 'Instance of class:', self.__class__.__name__

    # __repr__ = __str__


print 'creating foo'

foo = Foo()
