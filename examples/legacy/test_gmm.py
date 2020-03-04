# from workflow import simple_gaussian_mixture_model
from workflow import simple_gaussian_mixture_model2

import numpy
import numpy.random

# n = 100000
# data = numpy.random.normal(loc=0.0, scale=1.0, size=n * 3)
# data = numpy.hstack((data, numpy.random.normal(loc=0.0, scale=0.1, size=n)))
# numpy.random.shuffle(data)
# 
# (var, pi), _, _ = simple_gaussian_mixture_model(data, n_components=2)
# 
# normal = lambda x, sigmasq: numpy.exp(-0.5 * x * x / sigmasq) / numpy.sqrt(2 * numpy.pi * sigmasq)
# 
# y = numpy.log(pi[0] * normal(data, var[0]) + pi[1] * normal(data, var[1]))
# print('estimated = {}'.format(y.sum() / len(data)))
# y = numpy.log(0.75 * normal(data, 1.0 ** 2) + 0.25 * normal(data, 0.1 ** 2))
# print('true = {}'.format(y.sum() / len(data)))
# 
# import matplotlib.pylab as plt
# plt.hist(data, bins=100, density=True)
# x = numpy.linspace(-3.0, +3.0, 101)
# y = pi[0] * normal(x, var[0]) + pi[1] * normal(x, var[1])
# plt.plot(x, y, 'k--')
# y = 0.75 * normal(x, 1.0 ** 2) + 0.25 * normal(x, 0.1 ** 2)
# plt.plot(x, y, 'r--')
# plt.show()

n = 100000
x = numpy.random.normal(loc=0.0, scale=1.0, size=n * 3)
y = numpy.random.normal(loc=0.0, scale=1.0, size=n * 3)
data = numpy.sqrt(x ** 2 + y ** 2)
x = numpy.random.normal(loc=0.0, scale=0.1, size=n)
y = numpy.random.normal(loc=0.0, scale=0.1, size=n)
data = numpy.hstack((data, numpy.sqrt(x ** 2 + y ** 2)))
numpy.random.shuffle(data)

(var, pi), _, _ = simple_gaussian_mixture_model2(data, n_components=2)
print(var, pi)

p = lambda x, sigmasq: numpy.exp(-0.5 * x * x / sigmasq) * x / sigmasq

# y = numpy.log(pi[0] * normal(data, var[0]) + pi[1] * normal(data, var[1]))
# print('estimated = {}'.format(y.sum() / len(data)))
# y = numpy.log(0.75 * normal(data, 1.0 ** 2) + 0.25 * normal(data, 0.1 ** 2))
# print('true = {}'.format(y.sum() / len(data)))

import matplotlib.pylab as plt
n, bins, patches = plt.hist(data, bins=100, density=True)
x = (bins[: -1] + bins[1: ]) * 0.5
y = 0.75 * p(x, 1.0 ** 2) + 0.25 * p(x, 0.1 ** 2)
plt.plot(x, y, 'k--')
y = sum(pi_k * p(x, var_k) for var_k, pi_k in zip(var, pi))
plt.plot(x, y, 'r--')
plt.show()
