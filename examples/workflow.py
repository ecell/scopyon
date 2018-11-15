import luigi
import luigi.mock
import luigi.format
import luigi.contrib.external_program

import os.path
import hashlib
import numpy

import scopyon.samples
import scopyon.config
import scopyon.epifm
import scopyon.spot_detection


lower = numpy.array([3.26599e-08, 0.0, 0.0])
upper = numpy.array([3.26599e-08, 50033.2e-9, 50060e-9])
lengths = upper + lower
D1, D2, D3 = 0.222e-12, 0.032e-12, 0.008e-12
N = lengths[1] * lengths[2] * 1000 / (30e-6 * 30e-6)
N1, N2, N3 = int(0.16 * N), int(0.42 * N), int(0.42 * N)
N = (N1, N2, N3)
D = [(0, D1, D1), (0, D2, D2), (0, D3, D3)]

exposure_time = 30e-3
t0, t1 = 0.0, exposure_time * 150
cmin, cmax = 1900, 2500
low, high = 0, 255
ndiv = 5
dt = exposure_time / ndiv

HASH_MD5 = str(hashlib.md5(open(__file__).read().encode('utf-8')).hexdigest())
# HASH_MD5 = "39341abc9af514efbda7321570549a2b"
pathto = os.path.abspath(os.path.join("scopyon-images", HASH_MD5))

rndseed = 0
rng = numpy.random.RandomState(rndseed)

class GeneratePoints(luigi.Task):

    i = luigi.IntParameter(default=0)

    def requires(self):
        return GeneratePoints(i=self.i - 1) if self.i > 0 else ()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'points_{:07d}.npz'.format(self.i)), format=luigi.format.Nop)
        # return luigi.mock.MockTarget(self.__class__.__name__, format=luigi.format.Nop)

    def run(self):
        if self.i == 0:
            print(dir(self))
            points, start = scopyon.samples.generate_points(rng, lower=lower, upper=upper, N=N)
            t = t0
        else:
            input_ = self.input()
            with input_.open('r') as f:
                data = numpy.load(f)
                t, points = data['t'], data['points']
            points = scopyon.samples.move_points(rng, points, D, dt, lengths)
            t += dt

        output_ = self.output()
        with output_.open('w') as f:
            numpy.savez(f, t=t, points=points)

class OutputFrame(luigi.Task):

    i = luigi.IntParameter(default=0)

    def requires(self):
        imin, imax = self.i * ndiv, (self.i + 1) * ndiv
        return [GeneratePoints(i=i) for i in range(imin, imax)]

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'image_{:07d}.npz'.format(self.i)), format=luigi.format.Nop)

    def run(self):
        # import os
        # print('pid={}'.format(os.getpid()))

        config = scopyon.config.Config()

        ## set epifm configurations
        config.set_epifm_shutter(start_time=t0, end_time=t1)
        config.set_epifm_light_source(source_type='LASER', wave_length=512, flux_density=40 * 10, angle=72)
        config.set_epifm_fluorophore(
            fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0, radius=20)
        config.set_epifm_dichroic_mirror('FF493_574-Di01-25x36')
        focal_point = numpy.array([lengths[0] / 1e-9 * 0.0, lengths[1] / 1e-9 * 0.5, lengths[2] / 1e-9 * 0.5])
        config.set_epifm_magnification(magnification=60 * 1.5 * 4)
        config.set_epifm_detector(
            detector='EMCCD', image_size=(512, 512), pixel_length=16e-6, exposure_time=exposure_time,
            focal_point=focal_point, QE=0.92, readout_noise=100, emgain=300)
        config.set_epifm_analog_to_digital_converter(bit=16, offset=2000, fullwell=800000)

        ## set effects configurations
        config.set_effects_background(mean=0.01)
        config.set_effects_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
        config.set_effects_photobleaching(tau0=2.27, alpha=0.73)

        sim = scopyon.epifm.EPIFMSimulator(config, rng=rng)

        input_data = []
        for input_ in self.input():
            with input_.open('r') as f:
                data = numpy.load(f)
                input_data.append((data['t'], data['points']))

        camera, true_data = sim.output_frame(input_data, frame_index=self.i, rng=rng)

        output_ = self.output()
        with output_.open('w') as f:
            numpy.savez(f, camera=camera, true_data=true_data)

class Save8bitImage(luigi.Task):

    i = luigi.IntParameter(default=0)

    def requires(self):
        return OutputFrame(i=self.i)

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'image_{:07d}.png'.format(self.i)), format=luigi.format.Nop)

    def run(self):
        input_ = self.input()
        with input_.open('r') as f:
            data = numpy.load(f)
            camera = data['camera']

        bytedata = scopyon.image.convert_8bit(camera[: , : , 1], cmin, cmax, low, high)
        output_ = self.output()
        with output_.open('w') as f:
            scopyon.image.save_image(f, bytedata, low=low, high=high)

class GenerateAVI(luigi.Task):

    start = luigi.IntParameter(default=0)
    stop = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'output_{}_{}.avi'.format(self.start, self.stop)))

    def requires(self):
        return [Save8bitImage(i=i) for i in range(self.start, self.stop)]

    def run(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pylab as plt
        import matplotlib.animation as animation
        from matplotlib import cm
        cmap = cm.gray
        dpi = 100
        repeat = 1
        fps = 15

        writer = animation.writers['ffmpeg'](fps=fps)
        fig = plt.figure()
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        for input_ in self.input():
            data = plt.imread(input_.path)
            m, n, _ = data.shape
            fig.set_size_inches((m / dpi, n / dpi))
            break
        fig.add_axes(ax)
        with writer.saving(fig, self.output().path, dpi):
            for input_ in self.input():
                data = plt.imread(input_.path)
                m, n, _ = data.shape
                ax.imshow(data, interpolation='none', cmap=cmap, vmin=low, vmax=high)
                for _ in range(repeat):
                    writer.grab_frame()

class SpotDetection(luigi.Task):

    i = luigi.IntParameter(default=0)

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'spots_{:07d}.npy'.format(self.i)))

    def requires(self):
        return OutputFrame(i=self.i)

    def run(self):
        camera = numpy.load(self.input().path)['camera']
        spots = scopyon.spot_detection.spot_detection(
            camera[:, :, 1], min_sigma=2, max_sigma=3, num_sigma=20, threshold=40, overlap=0.5, opt=0)
        numpy.save(self.output().path, spots)


def main():
    import glob
    import os

    if not os.path.isdir(pathto):
        os.makedirs(pathto)

    luigi.build([GenerateAVI(stop=150)] + [SpotDetection(i=i) for i in range(150)], workers=1, local_scheduler=True)


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
