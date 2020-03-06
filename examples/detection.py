import scopyon
import numpy.random


config = scopyon.DefaultConfiguration()
config.default.detector.exposure_time = 33.0e-3
pixel_length = config.default.detector.pixel_length / config.default.magnification
L_2 = config.default.detector.image_size[0] * pixel_length * 0.5

rng = numpy.random.RandomState(123)
N = 100
inputs = rng.uniform(-L_2, +L_2, size=(N, 2))

img, infodict = scopyon.form_image(inputs, config=config, rng=rng, full_output=True)

spots = scopyon.analysis.spot_detection(
        img.as_array(), min_sigma=1, max_sigma=4, threshold=50.0, overlap=0.5)

r = 6
shapes = [dict(x=data[2], y=data[3], sigma=r, color='green')
        for data in infodict['true_data'].values()]
shapes += [dict(x=spot[0], y=spot[1], sigma=r, color='red')
        for spot in spots]
img.save("detection_000.png", shapes=shapes)
