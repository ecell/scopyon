import scopyon
import numpy.random


config = scopyon.DefaultConfiguration()
config.update("""
default:
    magnification: 360
    detector:
        exposure_time: 0.033
    effects:
        photo_bleaching:
            switch: true
            half_life: 2.5
""")
pixel_length = config.default.detector.pixel_length / config.default.magnification
L_2 = config.default.detector.image_size[0] * pixel_length * 0.5

rng = numpy.random.RandomState(123)
num_frames = 30
N = 1000
D = 0.1e-12  # m ** 2 / s
dt = config.default.detector.exposure_time
t = numpy.arange(0, (num_frames + 1) * dt, dt)
inputs = scopyon.generate_inputs(t, N=N, lower=-L_2, upper=+L_2, ndim=2, D=D, rng=rng)

img = list(scopyon.generate_images(inputs, num_frames=num_frames, config=config, rng=rng))
scopyon.save_video('video.mp4', img)
