import luigi
import luigi.mock
import luigi.format
import luigi.contrib.external_program

import os.path
import hashlib
import numpy
import json

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

# HASH_MD5 = str(hashlib.md5(open(__file__).read().encode('utf-8')).hexdigest())
# HASH_MD5 = "39341abc9af514efbda7321570549a2b"
HASH_MD5 = "ef4d8d76a2098fa811fa0c316568d5cf"
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
            points, _ = scopyon.samples.generate_points(rng, lower=lower, upper=upper, N=N)
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
        focal_point = numpy.array([lengths[0] * 0.0, lengths[1] * 0.5, lengths[2] * 0.5])
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
            camera[:, :, 1], min_sigma=2, max_sigma=3, num_sigma=20, threshold=40, overlap=0.5, opt=1)
        numpy.save(self.output().path, spots)

class CalculateDisplacement(luigi.Task):

    start = luigi.IntParameter(default=0)
    stop = luigi.IntParameter()
    frame_shift = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'dist_{}_{}_{}.png'.format(self.start, self.stop, self.frame_shift)))

    def requires(self):
        return [GeneratePoints(i=i) for i in range(self.start, self.stop)]

    def run(self):
        displacements = []

        for i in range(self.start, self.stop - self.frame_shift):
            data1 = numpy.load(self.input()[i].path)
            data2 = numpy.load(self.input()[i + self.frame_shift].path)

            # dt = data2['t'] - data1['t']
            data = data2['points'][: , : 3] - data1['points'][: , : 3]
            for dim in range(3):
                data.T[dim][data.T[dim] >= 0.5 * lengths[dim]] -= lengths[dim]
                data.T[dim][data.T[dim] <= -0.5 * lengths[dim]] += lengths[dim]
                # if lengths[dim] > 0.0:
                #     data.T[dim] /= lengths[dim]
            # displacements.append(data / (2 * dt))
            displacements.append(data)

        displacements = numpy.vstack(displacements)

        square_displacements = numpy.sqrt(displacements.T[1] ** 2 + displacements.T[2] ** 2)

        D_mean = (D1 * N1 + D2 * N2 + D3 * N3) / (N1 + N2 + N3)
        duration = self.frame_shift * dt

        x = numpy.linspace(0, numpy.sqrt(2 * D_mean * duration) * 5, 201)
        pdf = lambda r, sigma: r / (sigma * sigma) * numpy.exp(-r * r / (2 * sigma * sigma))
        y = [pdf(x, numpy.sqrt(2 * D_ * duration)) for D_ in (D1, D2, D3)]
        y = (y[0] * N1 + y[1] * N2 + y[2] * N3) / (N1 + N2 + N3)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pylab as plt
        fig, ax = plt.subplots(1, 1)
        ax.hist(square_displacements, bins=50, density=True, alpha=0.5, label='Observed')
        ax.plot(x, y, 'k--')
        plt.legend(loc='best')
        plt.savefig(self.output().path)

def simple_gaussian_mixture_model(x, n_components=3, max_iter=100, tol=1e-12):
    x = numpy.array(x)
    xsq = x ** 2
    var = xsq.sum() / len(x)

    # var = numpy.ones(n_components, dtype=numpy.float64) * var
    var = numpy.random.uniform(size=n_components) * var
    pi = numpy.random.uniform(size=n_components)
    pi /= pi.sum()

    normal = lambda xsq, sigmasq: numpy.exp(-0.5 * xsq / sigmasq) / numpy.sqrt(2 * numpy.pi * sigmasq)

    likelihood_old = 0.0
    for _ in range(max_iter):
        gamma = numpy.zeros((n_components, len(x)), dtype=numpy.float64)
        N = numpy.zeros(n_components, dtype=numpy.float64)

        print("[{}] var = {}, pi = {}".format(_, var, pi))
        for k in range(n_components):
            gamma[k] = pi[k] * normal(xsq, var[k])
        gamma_tot = gamma.sum(axis=0)

        assert numpy.all(gamma_tot > 0)
        likelihood = numpy.log(gamma_tot).sum() / len(x)
        print("[{}] log-likelihood = {}".format(_, likelihood))

        assert likelihood != 0
        # if likelihood >= likelihood_old and abs((likelihood - likelihood_old) / likelihood) < tol:
        #     break
        likelihood_old = likelihood

        for k in range(n_components):
            gamma[k] /= gamma_tot
            N[k] = gamma[k].sum()

        pi = N / N.sum()
        for k in range(n_components):
            var[k] = (gamma[k] * xsq).sum() / N[k]

    bic = -2 * likelihood_old * len(x) + n_components * numpy.log(len(x))
    print("BIC = {}".format(bic))
    aic = -2 * likelihood_old * len(x) + 2 * n_components
    print("AIC = {}".format(aic))
    return (var, pi), likelihood_old, (bic, aic)

def simple_gaussian_mixture_model2(x, n_components=3, max_iter=100, tol=1e-12):
    x = numpy.array(x)

    var = numpy.random.uniform(size=n_components) * (x.sum() / len(x))
    pi = numpy.ones(n_components, dtype=numpy.float64)
    pi /= pi.sum()

    normal = lambda x, sigmasq: numpy.exp(-0.5 * x * x / sigmasq) * x / sigmasq

    likelihood_old = 0.0
    for _ in range(max_iter):
        gamma = numpy.zeros((n_components, len(x)), dtype=numpy.float64)
        N = numpy.zeros(n_components, dtype=numpy.float64)

        print("[{}] var = {}, pi = {}".format(_, var, pi))
        for k in range(n_components):
            gamma[k] = pi[k] * normal(x, var[k])
        gamma_tot = gamma.sum(axis=0)

        if not numpy.all(gamma_tot > 0):
            return simple_gaussian_mixture_model2(x[gamma_tot > 0], n_components, max_iter, tol)
        assert numpy.all(gamma_tot > 0)

        likelihood = numpy.log(gamma_tot).sum() / len(x)
        print("[{}] log-likelihood = {}".format(_, likelihood))

        assert likelihood != 0
        if likelihood >= likelihood_old and abs((likelihood - likelihood_old) / likelihood) < tol:
            break
        likelihood_old = likelihood

        for k in range(n_components):
            gamma[k] /= gamma_tot
            N[k] = gamma[k].sum()

        pi = N / N.sum()
        for k in range(n_components):
            var[k] = 0.5 * (gamma[k] * x * x).sum() / N[k]

    bic = -2 * likelihood_old * len(x) + n_components * numpy.log(len(x))
    print("BIC = {}".format(bic))
    aic = -2 * likelihood_old * len(x) + 2 * n_components
    print("AIC = {}".format(aic))
    return (var, pi), likelihood_old, (bic, aic)

class CalculateDisplacementGMM(luigi.Task):

    start = luigi.IntParameter(default=0)
    stop = luigi.IntParameter()
    frame_shift = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'dist_{}_{}_{}.json'.format(self.start, self.stop, self.frame_shift)))

    def requires(self):
        return [GeneratePoints(i=i) for i in range(self.start, self.stop)]

    def run(self):
        displacements = []

        for i in range(self.start, self.stop - self.frame_shift):
            data1 = numpy.load(self.input()[i].path)
            data2 = numpy.load(self.input()[i + self.frame_shift].path)

            data = data2['points'][: , : 3] - data1['points'][: , : 3]
            for dim in range(3):
                data.T[dim][data.T[dim] >= 0.5 * lengths[dim]] -= lengths[dim]
                data.T[dim][data.T[dim] <= -0.5 * lengths[dim]] += lengths[dim]

            displacements.extend(data.T[1])
            displacements.extend(data.T[2])

        duration = self.frame_shift * dt
        result = {}
        result['true'] = {'D': (D1, D2, D3), 'pi': (N1 / (N1 + N2 + N3), N2 / (N1 + N2 + N3), N3 / (N1 + N2 + N3)), 'D_mean': (D1 * N1 + D2 * N2 + D3 * N3) / (N1 + N2 + N3)}
        result['estimated'] = []
        for n_components in range(1, 6):
            (var, pi), score, (bic, aic) = simple_gaussian_mixture_model(displacements, n_components=n_components, max_iter=1000)
            D = var / (2 * duration)
            result['estimated'].append({'D': tuple(D), 'pi': tuple(pi), 'D_mean': (D * pi).sum(), 'score': score, 'BIC': bic, 'AIC': aic, 'n_components': n_components})

        output_ = self.output()
        with output_.open('w') as f:
            json.dump(result, f)

class ClosestSquareDisplacement(luigi.Task):

    start = luigi.IntParameter(default=0)
    stop = luigi.IntParameter()
    frame_shift = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'disp_{}_{}_{}.png'.format(self.start, self.stop, self.frame_shift)))

    def requires(self):
        return [SpotDetection(i=i) for i in range(self.start, self.stop)]

    def run(self):
        square_displacements = []
        for i in range(self.start, self.stop - self.frame_shift):
            spots1 = numpy.load(self.input()[i].path)
            spots2 = numpy.load(self.input()[i + self.frame_shift].path)
            for spot in spots1:
                (height, center_x, center_y, width_x, width_y, bg) = spot
                val = numpy.min((spots2.T[1] - center_x) ** 2 + (spots2.T[2] - center_y) ** 2)
                square_displacements.append(val)
        square_displacements = numpy.array(square_displacements)
        # numpy.save(self.output().path, square_displacements)

        image_magnification = 60 * 1.5 * 4
        pixel_length = 16e-6 / image_magnification
        print("average={} ({})".format(numpy.average(square_displacements) * pixel_length * pixel_length, 4 * (D1 * N1 + D2 * N2 + D3 * N3) / (N1 + N2 + N3) * self.frame_shift * exposure_time))

        square_displacements = numpy.sqrt(square_displacements)

        D_mean = (D1 * N1 + D2 * N2 + D3 * N3) / (N1 + N2 + N3)
        duration = self.frame_shift * exposure_time

        # x = numpy.linspace(0, numpy.sqrt(2 * D_mean * duration) / pixel_length * 5, 201)
        # pdf = lambda r, sigma: r / (sigma * sigma) * numpy.exp(-r * r / (2 * sigma * sigma))
        # y = [pdf(x, numpy.sqrt(2 * D_ * duration) / pixel_length) for D_ in (D1, D2, D3)]
        # y = (y[0] * N1 + y[1] * N2 + y[2] * N3) / (N1 + N2 + N3)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pylab as plt
        fig, ax = plt.subplots(1, 1)
        n, bins, patches = ax.hist(square_displacements, bins=50, density=True, alpha=0.5, label='Observed')
        bins = numpy.array(bins)

        print("sum(n) = {}".format(sum(n)))
        print("dbins = {} ({})".format(bins[1] - bins[0], sum(n) * (bins[1] - bins[0])))

        I = lambda r0, r1, sigma: (-numpy.exp(-r1**2/(2*sigma**2)))-(-numpy.exp(-r0**2/(2*sigma**2)))
        y = [I(bins[: -1], bins[1: ], numpy.sqrt(2 * D_ * duration) / pixel_length) for D_ in (D1, D2, D3)]
        y = (y[0] * N1 + y[1] * N2 + y[2] * N3) / (N1 + N2 + N3) / (bins[1] - bins[0])
        x = (bins[:-1] + bins[1:]) * 0.5

        ax.plot(x, y, 'k--')
        plt.legend(loc='best')
        plt.savefig(self.output().path)

class ClosestSquareDisplacementGMM(luigi.Task):

    start = luigi.IntParameter(default=0)
    stop = luigi.IntParameter()
    frame_shift = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'disp_{}_{}_{}.json'.format(self.start, self.stop, self.frame_shift)))

    def requires(self):
        return [SpotDetection(i=i) for i in range(self.start, self.stop)]

    def run(self):
        displacements = []
        for i in range(self.start, self.stop - self.frame_shift):
            spots1 = numpy.load(self.input()[i].path)
            spots2 = numpy.load(self.input()[i + self.frame_shift].path)
            for spot in spots1:
                (height, center_x, center_y, width_x, width_y, bg) = spot
                Lsq = (spots2.T[1] - center_x) ** 2 + (spots2.T[2] - center_y) ** 2
                idx = Lsq.argmin()
                displacements.append(spots2[idx][1] - center_x)
                displacements.append(spots2[idx][2] - center_y)
        image_magnification = 60 * 1.5 * 4
        pixel_length = 16e-6 / image_magnification
        duration = self.frame_shift * exposure_time

        result = {}
        result['true'] = {'D': (D1, D2, D3), 'pi': (N1 / (N1 + N2 + N3), N2 / (N1 + N2 + N3), N3 / (N1 + N2 + N3)), 'D_mean': (D1 * N1 + D2 * N2 + D3 * N3) / (N1 + N2 + N3)}
        result['estimated'] = []
        for n_components in [3]: # range(1, 6):
            (var, pi), score, (bic, aic) = simple_gaussian_mixture_model(displacements, n_components=n_components, max_iter=10000)
            D = var * pixel_length * pixel_length / (2 * duration)
            result['estimated'].append({'D': tuple(D), 'pi': tuple(pi), 'D_mean': (D * pi).sum(), 'score': score, 'BIC': bic, 'AIC': aic, 'n_components': n_components})

        output_ = self.output()
        with output_.open('w') as f:
            json.dump(result, f)

class ClosestSquareDisplacementGMMPlot(luigi.Task):

    start = luigi.IntParameter(default=0)
    stop = luigi.IntParameter()
    frame_shift = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'gmm_{}_{}_{}.png'.format(self.start, self.stop, self.frame_shift)))

    def requires(self):
        return [ClosestSquareDisplacementGMM(self.start, self.stop, self.frame_shift)] + [SpotDetection(i=i) for i in range(self.start, self.stop)]

    def run(self):
        input_ = self.input()[0]
        with input_.open() as f:
            result = json.load(f)
        result = result['estimated'][0]

        displacements = []
        for i in range(self.start, self.stop - self.frame_shift):
            spots1 = numpy.load(self.input()[1 + i].path)
            spots2 = numpy.load(self.input()[1 + i + self.frame_shift].path)
            for spot in spots1:
                (height, center_x, center_y, width_x, width_y, bg) = spot
                Lsq = (spots2.T[1] - center_x) ** 2 + (spots2.T[2] - center_y) ** 2
                idx = Lsq.argmin()
                displacements.append(spots2[idx][1] - center_x)
                displacements.append(spots2[idx][2] - center_y)

        image_magnification = 60 * 1.5 * 4
        pixel_length = 16e-6 / image_magnification
        duration = self.frame_shift * exposure_time

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pylab as plt
        fig, ax = plt.subplots(1, 1)
        n, bins, patches = ax.hist(displacements, bins=101, density=True, alpha=0.5, label='Observed')
        bins = numpy.array(bins)

        normal = lambda xsq, sigmasq: numpy.exp(-0.5 * xsq / sigmasq) / numpy.sqrt(2 * numpy.pi * sigmasq)
        x = (bins[:-1] + bins[1:]) * 0.5
        y = numpy.array([result['pi'][k] * normal(x * x, result['D'][k] * 2.0 * duration / (pixel_length * pixel_length)) for k in range(result['n_components'])])
        for k in range(len(y)):
            ax.plot(x, y[k], 'r--')
        ax.plot(x, y.sum(axis=0), 'k--')

        # I = lambda r0, r1, sigma: (-numpy.exp(-r1**2/(2*sigma**2)))-(-numpy.exp(-r0**2/(2*sigma**2)))
        # y = [I(bins[: -1], bins[1: ], numpy.sqrt(2 * D_ * duration) / pixel_length) for D_ in (D1, D2, D3)]
        # y = (y[0] * N1 + y[1] * N2 + y[2] * N3) / (N1 + N2 + N3) / (bins[1] - bins[0])
        plt.legend(loc='best')
        plt.savefig(self.output().path)

class ClosestSquareDisplacementGMM2(luigi.Task):

    start = luigi.IntParameter(default=0)
    stop = luigi.IntParameter()
    frame_shift = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'disp_{}_{}_{}.json'.format(self.start, self.stop, self.frame_shift)))

    def requires(self):
        return [SpotDetection(i=i) for i in range(self.start, self.stop)]

    def run(self):
        distances = []
        for i in range(self.start, self.stop - self.frame_shift):
            spots1 = numpy.load(self.input()[i].path)
            spots2 = numpy.load(self.input()[i + self.frame_shift].path)
            for spot in spots1:
                (height, center_x, center_y, width_x, width_y, bg) = spot
                Lsq = (spots2.T[1] - center_x) ** 2 + (spots2.T[2] - center_y) ** 2
                distances.append(numpy.sqrt(Lsq.min()))
        image_magnification = 60 * 1.5 * 4
        pixel_length = 16e-6 / image_magnification
        duration = self.frame_shift * exposure_time

        result = {}
        result['true'] = {'D': (D1, D2, D3), 'pi': (N1 / (N1 + N2 + N3), N2 / (N1 + N2 + N3), N3 / (N1 + N2 + N3)), 'D_mean': (D1 * N1 + D2 * N2 + D3 * N3) / (N1 + N2 + N3)}
        result['estimated'] = []
        for n_components in range(1, 6):
            (var, pi), score, (bic, aic) = simple_gaussian_mixture_model2(distances, n_components=n_components, max_iter=10000)
            D = var * pixel_length * pixel_length / (2 * duration)
            result['estimated'].append({'D': tuple(D), 'pi': tuple(pi), 'D_mean': (D * pi).sum(), 'score': score, 'BIC': bic, 'AIC': aic, 'n_components': n_components})

        output_ = self.output()
        with output_.open('w') as f:
            json.dump(result, f)

class ClosestSquareDisplacementGMMPlot2(luigi.Task):

    start = luigi.IntParameter(default=0)
    stop = luigi.IntParameter()
    frame_shift = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(pathto, 'gmm_{}_{}_{}.png'.format(self.start, self.stop, self.frame_shift)))

    def requires(self):
        return [ClosestSquareDisplacementGMM2(self.start, self.stop, self.frame_shift)] + [SpotDetection(i=i) for i in range(self.start, self.stop)]

    def run(self):
        input_ = self.input()[0]
        with input_.open() as f:
            result = json.load(f)

        distances = []
        for i in range(self.start, self.stop - self.frame_shift):
            spots1 = numpy.load(self.input()[1 + i].path)
            spots2 = numpy.load(self.input()[1 + i + self.frame_shift].path)
            for spot in spots1:
                (height, center_x, center_y, width_x, width_y, bg) = spot
                Lsq = (spots2.T[1] - center_x) ** 2 + (spots2.T[2] - center_y) ** 2
                distances.append(numpy.sqrt(Lsq.min()))

        image_magnification = 60 * 1.5 * 4
        pixel_length = 16e-6 / image_magnification
        duration = self.frame_shift * exposure_time

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pylab as plt
        fig, ax = plt.subplots(1, 1)
        n, bins, patches = ax.hist(distances, bins=101, density=True, alpha=0.5, label='Observed')
        bins = numpy.array(bins)

        normal = lambda x, sigmasq: numpy.exp(-0.5 * x * x / sigmasq) * x / sigmasq
        x = (bins[:-1] + bins[1:]) * 0.5
        r = result['estimated'][3 - 1]
        y = numpy.array([r['pi'][k] * normal(x, r['D'][k] * 2.0 * duration / (pixel_length * pixel_length)) for k in range(r['n_components'])])
        for k in range(len(y)):
            label = "D={:.2e}, pi={:3f}".format(r['D'][k], r['pi'][k])
            ax.plot(x, y[k], 'r--', label=label)
        ax.plot(x, y.sum(axis=0), 'k--')
        plt.legend(loc='best')

        ax = plt.axes([.55, .3, .3, .3], facecolor='y')
        y = numpy.array([r['BIC'] for r in result['estimated']])
        ax.bar([r['n_components'] for r in result['estimated']], y, tick_label=[str(r['n_components']) for r in result['estimated']], align='center')
        ax.set_ylim(ymin=y.min() - 0.2 * (y.max() - y.min()), ymax=y.max() + 0.2 * (y.max() - y.min()))
        ax.set_title('n_components={}'.format(result['estimated'][y.argmin()]['n_components']))

        plt.savefig(self.output().path)


def main():
    import os

    if not os.path.isdir(pathto):
        os.makedirs(pathto)

    luigi.build(
        [GenerateAVI(stop=150)]
        + [ClosestSquareDisplacement(stop=150, frame_shift=fs) for fs in (1, 5, 10, 30)]
        + [ClosestSquareDisplacementGMMPlot2(stop=150, frame_shift=fs) for fs in (1, 5, 10, 30)]
        + [CalculateDisplacement(stop=150 * ndiv, frame_shift=10 * ndiv)]
        + [CalculateDisplacementGMM(stop=150 * ndiv, frame_shift=10 * ndiv)],
        workers=1, local_scheduler=True)


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
