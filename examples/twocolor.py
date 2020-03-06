import scopyon
import numpy.random


def main():
    config = scopyon.DefaultConfiguration()
    pixel_length = config.default.detector.pixel_length / config.default.magnification
    L_2 = config.default.detector.image_size[0] * pixel_length * 0.5

    rng = numpy.random.RandomState(123)
    N = 250
    inputs = rng.uniform(-L_2, +L_2, size=(N, 2))

    img1 = scopyon.form_image(inputs[: 200], config=config, rng=rng)
    img2 = scopyon.form_image(inputs[150: ], config=config, rng=rng)
    img = scopyon.Image.RGB(red=img1, green=img2)
    img.save("twocolor_000.png")


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
