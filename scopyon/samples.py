import numpy
import copy
import numbers
import collections

from logging import getLogger
_log = getLogger(__name__)


def generate_points(rng, N=None, conc=None, lower=None, upper=None, start=0, fluorophore_id=0):
    """Generate points distributed uniformly.

    Args:
        rng (numpy.RandomState): A random number generator.
        N (int or list, optional): The number of points to be generated.
        conc (float or list, optional): The concentration of points.
            Either one of `N` or `conc` must be given.
        upper (Number or array, optional): An upper limit of the position. Defaults to 1.
        lower (Number or array, optional): A lower limit of the position. Defaults to 0.
        start (int, optional): The first index. Defaults to 0.
        fluorophore_id (int, optional): The fluorophore id. Defaults to 0.

    Returns:
        array: An array of points. Each point consists of a coordinate, an index,
            a serial and lot of a molecule ID, a fluorophore ID,
            p_state (defaults to 1) and cyc_id (defaults to `inf`).
        int: The sum of `start` and the number of points generated.

    """
    if N is None and conc is None:
        raise ValueError('Either one of N or conc must be given.')

    ndim = 3

    lower = (numpy.ones(ndim) * lower if isinstance(lower, numbers.Number)
        else numpy.array(lower) if lower is not None
        else numpy.zeros(ndim))
    upper = (numpy.ones(ndim) * upper if isinstance(upper, numbers.Number)
        else numpy.array(upper) if upper is not None
        else numpy.ones(ndim))

    assert len(lower) >= ndim and len(upper) >= ndim

    for dim in range(ndim):
        if lower[dim] > upper[dim]:
            lower[dim], upper[dim] = upper[dim], lower[dim]

    if N is None:
        lengths = upper - lower
        size = numpy.multiply.reduce(lengths[lengths != 0])
        if isinstance(conc, collections.Iterable):
            N_list = [rng.poisson(size * conc_) for conc_ in conc]
        else:
            N_list = [rng.poisson(size * conc)]
    elif not isinstance(N, collections.Iterable):
        N_list = [N]
    else:
        N_list = N
    N = sum(N_list)

    if N <= 0:
        return numpy.array([]), start

    ret = numpy.zeros((N, ndim + 5))
    for dim in range(ndim):
        if lower[dim] < upper[dim]:
            ret[: , dim] = rng.uniform(lower[dim], upper[dim], N)
        else:
            ret[: , dim] = lower[dim]
    ret[: , ndim + 0] = numpy.arange(start, start + N)  # molecule ID serial
    ret[: , ndim + 1] = 1.0  # molecule ID lot

    N_ = 0
    for i, n in enumerate(N_list):
        ret[N_: N_ + n, ndim + 2] = fluorophore_id + i  # fluorophore ID serial
        N_ += n
    assert N == N_

    ret[: , ndim + 3] = 1.0  # p_state
    ret[: , ndim + 4] = float('inf')  # cyc_id

    return ret, start + N

def move_points(rng, points, D, dt, lengths=None):
    """
    Move points based on the given diffusion constant.

    Args:
        rng (numpy.RandomState): A random number generator.
        points (array): Points.
        D (list): A list of diffusion constants for each fluorophore.
            Its element is a constant or an array contains constants along with each axis.
        dt (float): A step interval.
        lengths (Number or array, optional): An upper limit of the position.
            No upper limit as default.

    Returns:
        array: An array of points updated.

    """
    ndim = 3

    fluorophore_id_min = int(points[: , ndim + 2].min())
    fluorophore_id_max = int(points[: , ndim + 2].max())

    if not isinstance(D, collections.Iterable):
        raise TypeError("'D' must be a list of diffusion constants for each fluorophore.")

    D = [numpy.ones(ndim) * D_ if isinstance(D_, numbers.Number) else numpy.array(D_) for D_ in D]
    lengths = numpy.ones(ndim) * lengths if isinstance(lengths, numbers.Number) else numpy.array(lengths)

    ret = points.copy()
    for i, fluorophore_id in enumerate(ret[: , ndim + 2]):
        fluorophore_id = int(fluorophore_id)
        for dim in range(ndim):
            if D[fluorophore_id][dim] > 0:
                ret[i, dim] += rng.normal(0.0, 2 * D[fluorophore_id][dim] * dt)

    if lengths is not None:
        for dim in range(ndim):
            ret[: , dim][ret[: , dim] >= lengths[dim]] -= lengths[dim]
            ret[: , dim][ret[: , dim] < 0] += lengths[dim]
        assert all((ret[: , dim] < lengths[dim]) & (ret[: , dim] >= 0))

    return ret

def attempt_reactions(rng, points, dt, transitions=None, degradation=None, synthesis=None, lower=None, upper=None, start=0):
    """
    Apply first order reactions.

    Args:
        rng (numpy.RandomState): A random number generator.
        points (array): Points.
        dt (float): A step interval.
        transitions (matrix, optional): A matrix of kinetic rates.
            k[i, j] means the first order kinetic rate for the transition from i-th to j-th.
        degradation (array, optional): An array of degradation rates.
            The probability that the reaction happens durint the step is given as `1 - exp(-k * dt)`.
        synthesis (array, optional): An array of synthesis rates per unit volume.
        upper (Number or array, optional): An upper limit of the position. Defaults to 1.
        lower (Number or array, optional): A lower limit of the position. Defaults to 0.
        start (int, optional): The first index. Defaults to 0.

    Returns:
        array: An array of points updated.
        int: The sum of `start` and the number of points generated.

    """
    ndim = 3

    ret = points.copy()

    if transitions is not None:
        transitions = numpy.asarray(transitions)
        Pacc = 1.0 - numpy.exp(-transitions * dt)
        Pacc = numpy.add.accumulate(Pacc, 1)

        if any(Pacc[:, -1] > 1.0):
            raise ValueError('The acceptance probability exceeds 1.')

        n, m = Pacc.shape

        rnd = rng.uniform(size=len(ret))
        for i, fluorophore_id in enumerate(ret[: , ndim + 2]):
            fluorophore_id = int(fluorophore_id)
            assert 0 <= fluorophore_id < n
            j = numpy.searchsorted(Pacc[fluorophore_id], rnd[i])
            if j != m:
                ret[i, ndim + 2] = j

    if degradation is not None:
        degradation = numpy.asarray(degradation)
        Pacc = 1.0 - numpy.exp(-degradation * dt)
        Pacc = numpy.add.accumulate(Pacc)

        rnd = rng.uniform(size=len(ret))
        surv = numpy.array([rnd[i] > Pacc[int(fluorophore_id)] for i, fluorophore_id in enumerate(ret[: , ndim + 2])])
        ret = ret[surv]

    if synthesis is not None:
        synthesis = numpy.asarray(synthesis)
        ret_, start = generate_points(rng, conc=synthesis * dt, lower=lower, upper=upper, start=start)
        if len(ret_) > 0:
            ret = numpy.vstack((ret, ret_))

    return ret, start


if __name__ == "__main__":
    import numpy
    from .samples import generate_points, move_points, attempt_reactions


    rng = numpy.random.RandomState(0)
    lengths = numpy.array([2220.872e-9, 50033.2e-9, 50060e-9])
    dt = 33e-3
    N1 = 120
    N2 = 60
    D1 = 0.1e-6
    D2 = 0.01e-6
    k12 = k21 = -numpy.log(1.0 - 0.2) / dt
    kd = k21
    size = numpy.multiply.reduce(lengths[lengths != 0])
    ks = N2 / size * kd

    points, start = generate_points(rng, upper=lengths, N=[N1, N2])
    t = 0.0
    print(points)
    print(points[: , 3])
    print(points[: , 5])
    print(points.T[0].min(), points.T[0].max(), lengths[0])
    print(points.T[1].min(), points.T[1].max(), lengths[1])
    print(points.T[2].min(), points.T[2].max(), lengths[2])

    points = move_points(rng, points, [D1, D2], dt, lengths)
    points, start = attempt_reactions(rng, points, dt, transitions=[[0.0, k12], [k21, 0.0]], degradation=[0.0, kd], synthesis=[ks, 0.0], upper=lengths, start=start)
    t += dt
    print(points)
    print(points[: , 3])
    print(points[: , 5])
    print(points.T[0].min(), points.T[0].max(), lengths[0])
    print(points.T[1].min(), points.T[1].max(), lengths[1])
    print(points.T[2].min(), points.T[2].max(), lengths[2])

    # points, start = generate_points(rng, upper=lengths, N=[N1, N2])
    # t = 0.0
    # while t < dt * 100:
    #     points = move_points(rng, points, [D1, D2], dt, lengths)
    #     points, start = attempt_reactions(rng, points, dt, transitions=[[0.0, k12], [k21, 0.0]], degradation=[0.0, kd], synthesis=[ks, 0.0], upper=lengths, start=start)
    #     t += dt
    #     print(t, sum(points[: , 5] == 0), sum(points[: , 5] == 1))
