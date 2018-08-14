import os
import numpy

from scopyon.samples import generate_points, move_points, attempt_reactions

from scopyon.config import Config
from scopyon.epifm import EPIFMSimulator
from scopyon.image import convert_8bit, save_image


def test_samples():
    rndseed = 0
    rng = numpy.random.RandomState(rndseed)

    exposure_time = 33e-3
    cmin, cmax = 1900, 2500
    low, high = 0, 255

    config = Config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'defaults.ini'))
    config.set_epifm_shutter(start_time=0, end_time=exposure_time)
    config.set_epifm_detector(exposure_time=exposure_time)

    sim = EPIFMSimulator(config, rng=rng)

    lower = numpy.array([3.26599e-08, 0.0, 0.0])
    upper = numpy.array([3.26599e-08, 50033.2e-9, 50060e-9])
    lengths = upper - lower
    dt = exposure_time
    N1 = 120
    N2 = 60
    D1 = 0.1e-6
    D2 = 0.01e-6
    k12 = k21 = -numpy.log(1.0 - 0.2) / dt
    kd = k21
    size = numpy.multiply.reduce(lengths[lengths != 0])
    ks = N2 / size * kd

    points, start = generate_points(rng, lower=lower, upper=upper, N=[N1, N2])
    t = 0.0
    input_data = [(t, points)]

    dt *= 0.1
    while t < dt * 10:
        points = move_points(rng, points, [(0, D1, D1), (0, D2, D2)], dt, lengths)
        # points, start = attempt_reactions(rng, points, dt, transitions=[[0.0, k12], [k21, 0.0]], degradation=[0.0, kd], synthesis=[ks, 0.0], upper=lengths, start=start)
        t += dt
        input_data.append((t, points))

    camera, true_data = sim.output_frame(input_data, 0, rng)
    bytedata = convert_8bit(camera[: , : , 1], cmin, cmax, low, high)
    # image_file_name = os.path.join(pathto, image_fmt % (frame_index))
    save_image('samples_%07d.png' % (0), bytedata, low=low, high=high)


if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(fmt)
    log = logging.getLogger('scopyon')
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    test_samples()
