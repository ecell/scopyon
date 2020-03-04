import warnings
import numbers
import collections.abc

import numpy
import numpy.random

from logging import getLogger
_log = getLogger(__name__)

__all__ = ["generate_inputs"]


def generate_points(rng, N=None, conc=None, lower=None, upper=None, start=0, ndim=3):
    """Generate points distributed uniformly.

    Args:
        rng (numpy.RandomState, optional): A random number generator.
        N (int or list, optional): The number of points to be generated.
        conc (float or list, optional): The concentration of points.
            Either one of `N` or `conc` must be given.
        upper (Number or array, optional): An upper limit of the position. Defaults to 1.
        lower (Number or array, optional): A lower limit of the position. Defaults to 0.
        start (int, optional): The first index. Defaults to 0.
        ndim (int, optional) The number of dimensions. Defaults to 3.

    Returns:
        A pair of an array and the last ID.

        An array of points. Each point consists of a coordinate, an index,
        p_state (defaults to 1) and cyc_id (defaults to `inf`).

        The last ID. The sum of `start` and the number of points generated.

    """
    if N is None and conc is None:
        raise ValueError('Either one of N or conc must be given.')

    _log.info('generate_points: N={}, conc={}, lower={}, upper={}, start={}, ndim={}.'.format(
        N, conc, lower, upper, start, ndim))

    lower = (numpy.ones(ndim) * lower if isinstance(lower, numbers.Number)
        else numpy.array(lower) if lower is not None
        else numpy.zeros(ndim))
    upper = (numpy.ones(ndim) * upper if isinstance(upper, numbers.Number)
        else numpy.array(upper) if upper is not None
        else numpy.ones(ndim))

    if len(lower) != ndim or len(upper) != ndim:
        raise ValueError(
            "The wrong size of limits was given [{},{} != {}].".format(len(lower), len(upper), ndim))

    for dim in range(ndim):
        if lower[dim] > upper[dim]:
            lower[dim], upper[dim] = upper[dim], lower[dim]

    _log.debug('lower was set to {}.'.format(lower))
    _log.debug('upper was set to {}.'.format(upper))

    if N is None:
        lengths = upper - lower
        size = numpy.prod(lengths[lengths != 0])
        if isinstance(conc, collections.abc.Iterable):
            N_list = [rng.poisson(size * conc_) for conc_ in conc]
        else:
            N_list = [rng.poisson(size * conc)]
    elif not isinstance(N, collections.abc.Iterable):
        N_list = [N]
    else:
        N_list = N
    N = sum(N_list)
    _log.debug('{} points would be distributed {}.'.format(N, list(N_list)))

    if N <= 0:
        return numpy.array([]), start

    ret = numpy.zeros((N, ndim + 2))
    for dim in range(ndim):
        if lower[dim] < upper[dim]:
            ret[: , dim] = rng.uniform(lower[dim], upper[dim], N)
        else:
            ret[: , dim] = lower[dim]
    ret[: , ndim + 0] = numpy.arange(start, start + N)  # Molecule ID
    ret[: , ndim + 1] = 1.0  # Photon state
    return ret, start + N

def move_points(rng, points, D, dt, ndim=3):
    """
    Move points based on the given diffusion constant.

    Args:
        rng (numpy.RandomState, optional): A random number generator.
        points (array): Points.
        D (float or array-like): Diffusion constants.
            It is a constant or an array contains constants along with each axis.
        dt (float): A step interval.
        ndim (int, optional) The number of dimensions. Defaults to 3.

    Returns:
        array: An array of points updated.

    """
    _log.info('move_points: D={}, dt={}, ndim={}.'.format(D, dt, ndim))
    _log.debug('{} points are given.'.format(len(points)))

    if D is None:
        D = numpy.zeros(ndim)
    elif not isinstance(D, collections.abc.Iterable):
        D = numpy.ones(ndim) * D
    else:
        D = numpy.asarray(D)
        assert len(D) == ndim

    ret = points.copy()
    for i in range(len(ret)):
        for dim in range(ndim):
            ret[i, dim] += rng.normal(0.0, numpy.sqrt(2 * D[dim] * dt))
    return ret

def generate_inputs(t, *, N=None, conc=None, lower=None, upper=None, D=None, ndim=3, rng=None):
    """Generate the input data.

    Args:
        t (arraylike): The time points.
        N (int or list, optional): The number of points to be generated.
        conc (float or list, optional): The concentration of points.
            Either one of `N` or `conc` must be given.
        upper (Number or array, optional): An upper limit of the position. Defaults to 1.
        lower (Number or array, optional): A lower limit of the position. Defaults to 0.
        D (float or array-like, optional): Diffusion constants.
            It is a constant or an array contains constants along with each axis.
            Defaults to 0.0.
        ndim (int, optional) The number of dimensions. Defaults to 3.
        rng (numpy.RandomState, optional): A random number generator.

    Returns:
        A pair of an array and the last ID.

        An array of points. Each point consists of a coordinate, an index,
        p_state (defaults to 1) and cyc_id (defaults to `inf`).

        The last ID. The sum of `start` and the number of points generated.

    """
    if rng is None:
        warnings.warn('A random number generator [rng] is not given.')
        rng = numpy.random.RandomState()

    t = sorted(t)
    points, _ = generate_points(rng, N=N, conc=conc, lower=lower, upper=upper, ndim=ndim)
    tcurrent = t[0]
    inputs = []
    for tnext in t:
        if tnext > tcurrent:
            points = move_points(rng, points, D=D, dt=(tnext - tcurrent), ndim=ndim)
            tcurrent = tnext
        else:
            points = points.copy()
        inputs.append((tcurrent, points))
    return inputs

# def attempt_reactions(rng, points, dt, transitions=None, degradation=None, synthesis=None, lower=None, upper=None, start=0, ndim=3):
#     """
#     Apply first order reactions.
# 
#     Args:
#         rng (numpy.RandomState): A random number generator.
#         points (array): Points.
#         dt (float): A step interval.
#         transitions (matrix, optional): A matrix of kinetic rates.
#             k[i, j] means the first order kinetic rate for the transition from i-th to j-th.
#         degradation (array, optional): An array of degradation rates.
#             The probability that the reaction happens durint the step is given as `1 - exp(-k * dt)`.
#         synthesis (array, optional): An array of synthesis rates per unit volume.
#         upper (Number or array, optional): An upper limit of the position. Defaults to 1.
#         lower (Number or array, optional): A lower limit of the position. Defaults to 0.
#         start (int, optional): The first index. Defaults to 0.
#         ndim (int, optional) The number of dimensions. Defaults to 3.
# 
#     Returns:
#         array: An array of points updated.
#         int: The sum of `start` and the number of points generated.
# 
#     """
#     _log.info('attempt_reactions: dt={}.'.format(dt))
#     _log.debug('{} points are given.'.format(len(points)))
# 
#     ndim = 3
# 
#     ret = points.copy()
# 
#     if transitions is not None:
#         transitions = numpy.asarray(transitions)
#         Pacc = 1.0 - numpy.exp(-transitions * dt)
#         Pacc = numpy.add.accumulate(Pacc, 1)
# 
#         if any(Pacc[:, -1] > 1.0):
#             raise ValueError('The acceptance probability exceeds 1.')
# 
#         n, m = Pacc.shape
# 
#         rnd = rng.uniform(size=len(ret))
#         cnt = 0
#         for i, fluorophore_id in enumerate(ret[: , ndim + 2]):
#             fluorophore_id = int(fluorophore_id)
#             if not (0 <= fluorophore_id < n):
#                 raise ValueError(
#                     "The invalid fluorophore id '{}' was given.".format(fluorophore_id))
#             j = numpy.searchsorted(Pacc[fluorophore_id], rnd[i])
#             if j != m:
#                 ret[i, ndim + 2] = j
#                 cnt += 1
#         _log.debug('{} transitions occur.'.format(cnt))
# 
#     if degradation is not None:
#         degradation = numpy.asarray(degradation)
#         Pacc = 1.0 - numpy.exp(-degradation * dt)
#         Pacc = numpy.add.accumulate(Pacc)
# 
#         rnd = rng.uniform(size=len(ret))
#         surv = numpy.array([rnd[i] > Pacc[int(fluorophore_id)] for i, fluorophore_id in enumerate(ret[: , ndim + 2])])
#         ret = ret[surv]
#         _log.debug('{} points were degradated.'.format(len(ret) - len(points)))
# 
#     if synthesis is not None:
#         synthesis = numpy.asarray(synthesis)
#         ret_, start = generate_points(rng, conc=synthesis * dt, lower=lower, upper=upper, start=start)
#         if len(ret_) > 0:
#             ret = numpy.vstack((ret, ret_))
#             _log.debug('{} points were born.'.format(len(ret_)))
# 
#     return ret, start
# 
# 
# if __name__ == "__main__":
#     from .samples import generate_points, move_points, attempt_reactions
# 
# 
#     rng = numpy.random.RandomState(0)
#     lengths = numpy.array([2220.872e-9, 50033.2e-9, 50060e-9])
#     dt = 33e-3
#     N1 = 120
#     N2 = 60
#     D1 = 0.1e-6
#     D2 = 0.01e-6
#     k12 = k21 = -numpy.log(1.0 - 0.2) / dt
#     kd = k21
#     size = numpy.multiply.reduce(lengths[lengths != 0])
#     ks = N2 / size * kd
# 
#     points, start = generate_points(rng, upper=lengths, N=[N1, N2])
#     t = 0.0
#     print(points)
#     print(points[: , 3])
#     print(points[: , 5])
#     print(points.T[0].min(), points.T[0].max(), lengths[0])
#     print(points.T[1].min(), points.T[1].max(), lengths[1])
#     print(points.T[2].min(), points.T[2].max(), lengths[2])
# 
#     points = move_points(rng, points, [D1, D2], dt, lengths)
#     points, start = attempt_reactions(rng, points, dt, transitions=[[0.0, k12], [k21, 0.0]], degradation=[0.0, kd], synthesis=[ks, 0.0], upper=lengths, start=start)
#     t += dt
#     print(points)
#     print(points[: , 3])
#     print(points[: , 5])
#     print(points.T[0].min(), points.T[0].max(), lengths[0])
#     print(points.T[1].min(), points.T[1].max(), lengths[1])
#     print(points.T[2].min(), points.T[2].max(), lengths[2])
# 
#     # points, start = generate_points(rng, upper=lengths, N=[N1, N2])
#     # t = 0.0
#     # while t < dt * 100:
#     #     points = move_points(rng, points, [D1, D2], dt, lengths)
#     #     points, start = attempt_reactions(rng, points, dt, transitions=[[0.0, k12], [k21, 0.0]], degradation=[0.0, kd], synthesis=[ks, 0.0], upper=lengths, start=start)
#     #     t += dt
#     #     print(t, sum(points[: , 5] == 0), sum(points[: , 5] == 1))
