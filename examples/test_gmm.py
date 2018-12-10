#from . import example.workflow
from workflow import simple_gaussian_mixture_model

import numpy
import numpy.random

n = 100000
data = numpy.random.normal(loc=0.0, scale=1.0, size=n * 3)
data = numpy.hstack((data, numpy.random.normal(loc=0.0, scale=0.1, size=n)))
numpy.random.shuffle(data)

(var, pi), _, _ = simple_gaussian_mixture_model(data, n_components=2)

normal = lambda x, sigmasq: numpy.exp(-0.5 * x * x / sigmasq) / numpy.sqrt(2 * numpy.pi * sigmasq)

y = numpy.log(pi[0] * normal(data, var[0]) + pi[1] * normal(data, var[1]))
print('estimated = {}'.format(y.sum() / len(data)))
y = numpy.log(0.75 * normal(data, 1.0 ** 2) + 0.25 * normal(data, 0.1 ** 2))
print('true = {}'.format(y.sum() / len(data)))

import matplotlib.pylab as plt
plt.hist(data, bins=100, density=True)
x = numpy.linspace(-3.0, +3.0, 101)
y = pi[0] * normal(x, var[0]) + pi[1] * normal(x, var[1])
plt.plot(x, y, 'k--')
y = 0.75 * normal(x, 1.0 ** 2) + 0.25 * normal(x, 0.1 ** 2)
plt.plot(x, y, 'r--')
plt.show()
