import scopyon
import numpy.random


config = scopyon.DefaultConfiguration()
config.default.detector.exposure_time = 33.0e-3
pixel_length = config.default.detector.pixel_length / config.default.magnification
L_2 = config.default.detector.image_size[0] * pixel_length * 0.5

rng = numpy.random.RandomState(123)
N = 100
inputs = rng.uniform(-L_2, +L_2, size=(N, 2))

img = scopyon.form_image(inputs, config=config, rng=rng)
img.save("tirf_000.png")
