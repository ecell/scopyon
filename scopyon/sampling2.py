import numbers
import collections.abc
import warnings

import numpy
import numpy.random
import numpy.lib.recfunctions

from logging import getLogger
_log = getLogger(__name__)

__all__ = ["sample"]


def __generate_points(rng, N, lower, upper, ndim):
    ret = numpy.zeros((sum(N), ndim + 2))

    _log.debug('{} points would be distributed.'.format(ret.shape[0]))

    tot = 0
    for i, cnt in enumerate(N):
        for dim in range(ndim):
            if lower[dim] < upper[dim]:
                ret[tot: tot + cnt, dim] = rng.uniform(lower[dim], upper[dim], cnt)
            else:
                ret[tot: tot + cnt, dim] = lower[dim]
        ret[tot: tot + cnt, ndim + 0] = i
        tot += cnt
    start = 0
    ret[: , ndim + 1] = numpy.arange(start, start + ret.shape[0])  # Molecule ID
    return ret

def __move_points(rng, points, D, lower, upper, dt, ndim):
    _log.info('move_points: D={}, dt={}, ndim={}.'.format(D, dt, ndim))
    _log.debug('{} points are given.'.format(points.shape[0]))

    ret = points.copy()
    for i in range(len(ret)):
        state = int(ret[i, ndim + 0])
        scale = numpy.sqrt(2 * D[state] * dt)
        for dim in range(ndim):
            ret[i, dim] += rng.normal(0.0, scale)
    for dim in range(ndim):
        if upper[dim] > lower[dim]:
            ret[: , dim] = (
                (ret[:, dim] - lower[dim]) % (upper[dim] - lower[dim]) + lower[dim])
        else:
            ret[:, dim] = lower[dim]
    return ret

def __transition_states(rng, points, transmat, dt, ndim):
    _log.info('transition_states: dt={}, ndim={}.'.format(dt, ndim))
    _log.debug('{} points are given.'.format(points.shape[0]))

    n, m = transmat.shape
    P = 1 - numpy.exp(-transmat * dt)
    assert (P.sum(axis=1) <= 1.0).all()
    P.ravel()[: : n + 1] = 1.0 - P.sum(axis=1)
    Pacc = P.cumsum(axis=1)

    ret = points.copy()
    for i in range(ret.shape[0]):
        state = int(ret[i, ndim + 0])
        rnd = rng.uniform(0, 1)
        state_next = numpy.searchsorted(Pacc[state], rnd, side='leff')
        ret[i, ndim + 0] = state_next
    return ret

def sample(t, N, *, lower=None, upper=None, D=None, transmat=None, ndim=3, rng=None):
    """Generate the points.

    Args:
        t (arraylike): The time points.
        N (int or list): The initial number of points for each state.
        upper (Number or array, optional): An upper limit of the position. Defaults to 1.
        lower (Number or array, optional): A lower limit of the position. Defaults to 0.
        D (float or array-like, optional): Diffusion constants.
            It is a constant or an array for each state.
            Defaults to 0.0.
        transmat (array-like, optional): A state transition rate matrix.
            It must be a square matrix of size n, where n is the number of states.
        ndim (int, optional) The number of dimensions. Defaults to 3.
        rng (numpy.RandomState, optional): A random number generator.

    Returns:
        A list of points at each time point.
        An array of points. Each point consists of a coordinate, state, and its index.

    """
    if not isinstance(N, collections.abc.Iterable):
        N = [N]
    else:
        N = N

    lower = (numpy.ones(ndim) * lower if isinstance(lower, numbers.Number)
        else numpy.array(lower) if lower is not None
        else numpy.zeros(ndim))
    upper = (numpy.ones(ndim) * upper if isinstance(upper, numbers.Number)
        else numpy.array(upper) if upper is not None
        else numpy.ones(ndim))
    if len(lower) < ndim or len(upper) < ndim:
        raise ValueError(
            "The wrong size of limits was given"
            " [(lower={}, upper={}) != {}].".format(len(lower), len(upper), ndim))
    for dim in range(ndim):
        if lower[dim] > upper[dim]:
            lower[dim], upper[dim] = upper[dim], lower[dim]
    _log.debug('lower was set to {}.'.format(lower))
    _log.debug('upper was set to {}.'.format(upper))

    if D is None:
        D = numpy.zeros(len(N))
    elif not isinstance(D, collections.abc.Iterable):
        D = numpy.ones(len(N)) * D
    else:
        D = numpy.asarray(D)
        assert len(D) == len(N)

    if transmat is not None:
        transmat = numpy.asarray(transmat)
        n, m = transmat.shape
        assert n == m
        assert (transmat.diagonal() == 0).all()
        assert n == len(N)

    if rng is None:
        warnings.warn('A random number generator [rng] is not given.')
        rng = numpy.random.RandomState()

    points = __generate_points(rng, N=N, lower=lower, upper=upper, ndim=ndim)
    tcurrent = t[0]
    ret = [points.copy()]
    for tnext in t[1: ]:
        if tnext > tcurrent:
            dt = tnext - tcurrent
            points = __move_points(
                rng, points, D=D, lower=lower, upper=upper, dt=dt, ndim=ndim)
            if transmat is not None:
                points = __transition_states(
                    rng, points, transmat=transmat, dt=dt, ndim=ndim)
            tcurrent = tnext
        else:
            assert tnext == tcurrent
        ret.append(points)
    return ret


if __name__ == "__main__":
    ret = sample(
        [0.0, 1.0e-1, 2.0e-1, 3.0e-1], N=[5, 3, 2], D=[4e-3, 1e-3, 0.0],
        transmat=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ndim=2)
    for points in ret:
        print(points)
