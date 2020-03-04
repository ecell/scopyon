import scopyon.config
import scopyon.base

import numpy.random


def main():
    config = scopyon.config.DefaultConfiguration()
    config.default.detector.exposure_time = 33.0e-3

    pixel_length = config.default.detector.pixel_length / config.default.magnification
    L_2 = config.default.detector.image_size[0] * pixel_length * 0.5

    rng = numpy.random.RandomState(123)
    N = 100
    inputs = rng.uniform(-L_2, +L_2, size=(N, 2))
    img = scopyon.base.form_image(inputs, config=config, rng=rng)
    img.save("tirf_000.png")


if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(fmt)
    log = logging.getLogger('scopyon')
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    main()
