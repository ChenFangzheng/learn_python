edict = {}.fromkeys(('foo', 'bar'))
fdict = dict((['x', 1], ['y', 2]))

print edict, fdict

for key in fdict:
    print 'key=%s, value=%s' % (key, fdict[key])


print fdict.pop('x')

for key in fdict:
    print 'key=%s, value=%s' % (key, fdict[key])

edict.update(fdict)

print edict
