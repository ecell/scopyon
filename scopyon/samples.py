import numpy

from logging import getLogger
_log = getLogger(__name__)


def generate_points(rng, upper=None, N=None, conc=None, lower=None, ndim=3, start=0):
    if (N is None) == (conc is None):
        raise ValueError('Either N or conc must be given.')
    lower = lower if lower is not None else numpy.zeros(ndim)
    upper = upper if upper is not None else numpy.ones(ndim)
    for dim in range(ndim):
        if lower[dim] > upper[dim]:
            lower[dim], upper[dim] = upper[dim], lower[dim]
    lengths = upper - lower
    area = numpy.multiply.reduce(lengths[lengths != 0])
    if N is None:
        N = int(area * conc)
    if N <= 0:
        return numpy.array([])

    ret = []
    for i in range(N):
        coord = (
            rng.uniform(lower[0], upper[0]),
            rng.uniform(lower[1], upper[1]),
            rng.uniform(lower[2], upper[2]),
            )
        ret.append((coord, start + i, 1.0, 0.0, 1.0, float('inf')))
    # ret = numpy.zeros((N, ndim + 3))
    # for dim in range(ndim):
    #     if lower[dim] != upper[dim]:
    #         ret[: , dim] = rng.uniform(lower[dim], upper[dim], N)
    # ret[: , ndim + 0] = numpy.arange(start, start + N)
    # ret[: , ndim + 1] = numpy.ones(N)
    # ret[: , ndim + 2] = numpy.zeros(N)
    return ret
