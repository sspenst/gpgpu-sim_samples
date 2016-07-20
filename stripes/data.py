from random import randint

f = open('input', 'w')
depth = 3
neuron = 8
synapse = 4
filters = 2

# neuron
for k in xrange(depth):
	for i in xrange(neuron):
		for j in xrange(neuron):
			f.write(str(randint(0,9)) + ' ')
		f.write('\n')
	f.write('\n')
f.write('\n')

# synapse
for l in xrange(filters):
	for k in xrange(depth):
		for i in xrange(synapse):
			for j in xrange(synapse):
				f.write(str(randint(0,1)) + ' ')
			f.write('\n')
		f.write('\n')
	f.write('\n')
