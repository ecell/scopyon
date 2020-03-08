import copy
import math
import warnings
import os
import numbers
import functools
from multiprocessing import Pool

import numpy

import scipy.special
import scipy.integrate
from scipy.special import j0, i1e
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from . import io
from .image import Image
from . import constants

from logging import getLogger
_log = getLogger(__name__)


# def _polar2cartesian_coordinates(r, t, x, y):
#     X, Y = numpy.meshgrid(x, y)
#     new_r = numpy.sqrt(X * X + Y * Y)
#     new_t = numpy.arctan2(X, Y)
# 
#     ir = interp1d(r, numpy.arange(len(r)), bounds_error=False)
#     it = interp1d(t, numpy.arange(len(t)))
# 
#     new_ir = ir(new_r.ravel())
#     new_it = it(new_t.ravel())
#     new_ir[new_r.ravel() > r.max()] = len(r) - 1
#     new_ir[new_r.ravel() < r.min()] = 0
# 
#     return numpy.array([new_ir, new_it])
# 
# def _polar2cartesian(grid, coordinates, shape):
#     r = shape[0] - 1
#     psf_cart = numpy.empty([2 * r + 1, 2 * r + 1])
#     psf_cart[r: , r: ] = map_coordinates(grid, coordinates, order=0).reshape(shape)
#     psf_cart[r: , : r] = psf_cart[r: , : r: -1]
#     psf_cart[: r, : ] = psf_cart[: r: -1, : ]
#     return psf_cart

class PointSpreadingFunction:

    def __init__(self, psf_radial_cutoff, psf_radial_width, psf_depth_cutoff, fluorophore_type, psf_wavelength):
        # fluorophore
        self.fluorophore_type = fluorophore_type
        self.psf_wave_length = psf_wavelength
        self.psf_radial_width = psf_radial_width  #XXX: Only required for Gaussian

        self.__fluorophore_psf = {}
        self.__cutoff_radial = psf_radial_cutoff
        self.__cutoff_depth = psf_depth_cutoff
        self.__resolution_depth = 1e-9
        self.__resolution_radial = 1e-9

        if self.fluorophore_type == 'Gaussian':
            if self.psf_radial_width is None:
                raise ValueError(
                        'fluorophore.radial_width must be given for Gaussian type fluorophore.')
            self.get_distribution = self.get_gaussian_distribution
            #XXX: `partial` is not picklable.
            # self.get_distribution = functools.partial(
            #         self.__get_gaussian_distribution, radial_width=self.psf_radial_width)
        else:
            self.get_distribution = self.get_born_wolf_distribution
            #XXX: `partial` is not picklable.
            # self.get_distribution = functools.partial(
            #         self.__get_born_wolf_distribution_old, wave_length=self.psf_wave_length)

    def get(self, depth):
        depth = abs(depth)
        if depth < self.__cutoff_depth + self.__resolution_depth:
            assert depth >= 0
            key = int(depth / self.__resolution_depth)
            depth = key * self.__resolution_depth
        else:
            key = -1
            depth = self.__cutoff_depth

        if key not in self.__fluorophore_psf:
            self.__fluorophore_psf[key] = self.__get(depth)
        return self.__fluorophore_psf[key]

    def __get(self, depth):
        radial = numpy.arange(0.0, self.__cutoff_radial, self.__resolution_radial, dtype=float)
        r = radial / self.__resolution_radial
        radial_cutoff = self.__cutoff_radial / self.__resolution_radial
        psf_r = self.get_distribution(radial, depth)
        psf_cart = self.radial_to_cartesian(radial, psf_r, self.__cutoff_radial, self.__resolution_radial)
        return psf_cart

    @staticmethod
    def radial_to_cartesian(radial, radial_distribution, radial_cutoff, radial_resolution):
        # r = radial / radial_resolution
        # radial_cutoff = radial_cutoff / radial_resolution

        # theta = numpy.linspace(0, 90, 91)  #XXX: theta resolution
        # z = numpy.linspace(0, radial_cutoff, len(r))
        # y = numpy.linspace(0, radial_cutoff, len(r))

        # coordinates = _polar2cartesian_coordinates(r, theta, z, y)
        # theta = numpy.ones_like(theta)
        # polar_distribution = numpy.array(list(map(lambda x: theta * x, radial_distribution)))
        # return _polar2cartesian(polar_distribution, coordinates, (len(z), len(y)))

        assert isinstance(radial, numpy.ndarray) and radial.ndim == 1
        assert isinstance(radial_distribution, numpy.ndarray) and radial.ndim == 1
        assert radial.size == radial_distribution.size
        assert isinstance(radial_cutoff, numbers.Real)
        assert isinstance(radial_resolution, numbers.Real)

        shape = (2 * (radial.size  - 1) + 1, 2 * (radial.size  - 1) + 1)
        X, Y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
        X = (X.ravel() - (radial.size - 1)) * radial_resolution
        Y = (Y.ravel() - (radial.size - 1)) * radial_resolution
        R = numpy.sqrt(X ** 2 + Y ** 2)
        R[R > radial.max()] = radial.max()  #XXX
        interp_func = interp1d(radial, radial_distribution)
        P = interp_func(R)
        return P.reshape(shape)

    def get_gaussian_distribution(self, radial, depth):
        psf = self.__get_gaussian_distribution(radial, depth, self.psf_radial_width)
        return psf

    @staticmethod
    def __get_gaussian_distribution(r, _z, radial_width):
        return numpy.exp(-0.5 * (r / radial_width) ** 2) / (2 * numpy.pi * radial_width * radial_width)

        # #XXX: normalization means I in __get_normalization.
        # # For normalization
        # # norm = list(map(lambda x: True if x > 1e-4 else False, I))
        # norm = (normalization > 1e-4)
        # # Ir = sum(list(map(lambda x: x * numpy.exp(-0.5 * (r / self.psf_width[0]) ** 2), norm)))
        # # Iz = sum(list(map(lambda x: x * numpy.exp(-0.5 * (z / self.psf_width[1]) ** 2), norm)))
        # Ir = norm * numpy.exp(-0.5 * numpy.power(r / width[0], 2))
        # Iz = norm * numpy.exp(-0.5 * numpy.power(r / width[1], 2))
        # return numpy.array(list(map(lambda x: Ir * x, Iz)))

    def get_born_wolf_distribution(self, radial, depth):
        # psf = self.__get_born_wolf_distribution(radial, depth, self.psf_wave_length)
        psf = self.__get_born_wolf_distribution_old(radial, depth, self.psf_wave_length)
        # NA = 1.4  # self.objective_NA
        # k = 2.0 * numpy.pi / self.psf_wave_length
        # alpha = k * NA
        # unit_area = 1e-9 ** 2
        # psf *= 1.0 / (alpha * alpha / numpy.pi * unit_area)  #TODO: Unknown normalization factor
        # print("Normalization:", 1.0 / (alpha * alpha / numpy.pi * unit_area))
        return psf

    @staticmethod
    def __get_born_wolf_distribution(r, z, wave_length):
        assert isinstance(r, numpy.ndarray) and r.ndim == 1
        assert isinstance(z, numbers.Real) and z >= 0.0
        assert isinstance(wave_length, numbers.Real) and wave_length > 0

        # set Numerical Appature
        NA = 1.4  # self.objective_NA

        # set alpha and psi
        k = 2.0 * numpy.pi / wave_length
        alpha = k * NA
        psi = 0.5 * alpha * z * NA

        psf = numpy.zeros(r.size, dtype=r.dtype)
        for i in range(r.size):
            alpha_r = alpha * r[i]
            I_real, _ = scipy.integrate.quad(lambda rho: j0(alpha_r * rho) * +numpy.cos(psi * rho * rho) * rho, 0, 1.0)
            I_imag, _ = scipy.integrate.quad(lambda rho: j0(alpha_r * rho) * -numpy.sin(psi * rho * rho) * rho, 0, 1.0)
            psf[i] = I_real ** 2 + I_imag ** 2
        psf *= alpha * alpha / numpy.pi  #XXX: normalization
        return psf

    @staticmethod
    def __get_born_wolf_distribution_old(r, z, wave_length):
        # set Numerical Appature
        NA = 1.4  # self.objective_NA

        # set alpha and gamma consts
        k = 2.0 * numpy.pi / wave_length
        alpha = k * NA
        gamma = k * numpy.power(NA / 2, 2)

        # set rho parameters
        N = 100
        drho = 1.0 / N
        # rho = numpy.array([(i + 1) * drho for i in range(N)])
        rho = numpy.arange(1, N + 1) * drho
        # print(f'r.shape = {r.shape}')
        # print(f'rho.shape = {rho.shape}')

        J0 = numpy.array(list(map(lambda x: j0(x * alpha * rho), r)))
        # print(f'J0.shape = {J0.shape}')
        Y  = numpy.exp(-2 * 1.j * z * gamma * rho * rho) * rho * drho
        # print(f'Y.shape = {Y.shape}')
        I  = Y * J0
        # print(f'I.shape = {I.shape}')
        I_sum = I.sum(axis=1)
        # print(f'I_sum.shape = {I_sum.shape}')

        # set PSF
        psf = numpy.array(list(map(lambda x: abs(x) ** 2, I_sum)))
        # print(f'psf.shape = {psf.shape}')
        # print(f'psf.sum = {psf.sum()}')

        psf *= alpha * alpha / numpy.pi  #XXX: normalization
        return psf

    def overlay_signal(self, expected, p_i, pixel_length, normalization=1.0):
        # fluorophore axial position
        if normalization <= 0.0:
            return
        depth = p_i[0]
        signal = self.get(depth)
        self.overlay_signal_(expected, signal, p_i, pixel_length, self.__resolution_radial, normalization)

    @staticmethod
    def overlay_signal_(expected, signal, p_i, pixel_length, signal_resolution, normalization):
        _, xi, yi = p_i
        Nw_pixel, Nh_pixel = expected.shape
        # signal_resolution = 1.0e-9  # m  #=> self.__cutoff_radial / (len(r) - 1) ???
        signal_width = signal_resolution * (signal.shape[0] - 1)
        signal_height = signal_resolution * (signal.shape[1] - 1)
        expected_width = Nw_pixel * pixel_length  # Not `(Nw_pixel - 1) * pixel_length`
        expected_height = Nh_pixel * pixel_length  # Not `(Nh_pixel - 1) * pixel_length`

        imin = math.floor((expected_width * 0.5 + xi - signal_width * 0.5) / pixel_length)
        imax = math.ceil((expected_width * 0.5 + xi + signal_width * 0.5) / pixel_length)
        iarray = numpy.arange(max(0, imin), min(Nw_pixel, imax) + 1)
        left = (iarray * pixel_length - (expected_width * 0.5 + xi - signal_width * 0.5)) / signal_resolution
        left = numpy.ceil(left).astype(int)
        # assert left[1] > 0
        # if imin >= 0:
        #     assert left[0] <= 0
        #     left[0] = 0
        # else:
        #     assert left[0] > 0
        # assert left[-2] < signal.shape[0]
        # if imax <= Nw_pixel:
        #     assert left[-1] >= signal.shape[0]
        #     left[-1] = signal.shape[0]
        # else:
        #     assert left[-1] < signal.shape[0]
        left[0] = max(left[0], 0)
        left[-1] = min(left[-1], signal.shape[0])

        jmin = math.floor((expected_height * 0.5 + yi - signal_height * 0.5) / pixel_length)
        jmax = math.ceil((expected_height * 0.5 + yi + signal_height * 0.5) / pixel_length)
        jarray = numpy.arange(max(0, jmin), min(Nh_pixel, jmax) + 1)
        top = (jarray * pixel_length - (expected_height * 0.5 + yi - signal_height * 0.5)) / signal_resolution
        top = numpy.ceil(top).astype(int)
        # assert top[1] > 0
        # if jmin >= 0:
        #     assert top[0] <= 0
        #     top[0] = 0
        # else:
        #     assert top[0] > 0
        # assert top[-2] < signal.shape[1]
        # if jmax <= Nw_pixel:
        #     assert top[-1] >= signal.shape[1]
        #     top[-1] = signal.shape[1]
        # else:
        #     assert top[-1] < signal.shape[1]
        top[0] = max(top[0], 0)
        top[-1] = min(top[-1], signal.shape[1])

        unit_area = signal_resolution * signal_resolution
        for i, i0, i1 in zip(iarray, left[: -1], left[1: ]):
            for j, j0, j1 in zip(jarray, top[: -1], top[1: ]):
                photons = signal[i0: i1, j0: j1].sum() * unit_area
                if photons > 0:
                    expected[i, j] += photons * normalization

        # # particle position
        # _, yi, zi = p_i
        # dy = (yi - signal_resolution * signal.shape[0] / 2 + pixel_length * (Nw_pixel - 1) / 2) / pixel_length
        # dz = (zi - signal_resolution * signal.shape[1] / 2 + pixel_length * (Nh_pixel - 1) / 2) / pixel_length
        # imin = math.ceil(dy)
        # imax = math.ceil(signal.shape[0] / pixel_ratio + dy) - 1
        # jmin = math.ceil(dz)
        # jmax = math.ceil(signal.shape[1] / pixel_ratio + dz) - 1
        # imin = max(imin, 0)
        # imax = min(imax, Nw_pixel)
        # jmin = max(jmin, 0)
        # jmax = min(jmax, Nh_pixel)

        # iarray = numpy.arange(imin, imax + 1)
        # ibottom = (iarray[: -1] - dy) * pixel_ratio
        # ibottom = numpy.maximum(numpy.floor(ibottom).astype(int), 0)
        # itop = (iarray[1: ] - dy) * pixel_ratio
        # itop = numpy.minimum(numpy.floor(itop).astype(int), signal.shape[0])
        # irange = numpy.vstack((iarray[: -1], ibottom, itop)).T
        # irange = irange[irange[:, 2] > irange[:, 1]]

        # jarray = numpy.arange(jmin, jmax)
        # jbottom = (jarray[: -1] - dz) * pixel_ratio
        # jbottom = numpy.maximum(numpy.floor(jbottom).astype(int), 0)
        # jtop = (jarray[1: ] - dz) * pixel_ratio
        # jtop = numpy.minimum(numpy.floor(jtop).astype(int), signal.shape[1])
        # jrange = numpy.vstack((jarray[: -1], jbottom, jtop)).T
        # jrange = jrange[jrange[:, 2] > jrange[:, 1]]

        # unit_area = signal_resolution * signal_resolution
        # for i, i0, i1 in irange:
        #     for j, j0, j1 in jrange:
        #         photons = signal[i0: i1, j0: j1].sum() * unit_area
        #         if photons > 0:
        #             expected[i, j] += photons * normalization

def cylindrical_coordinate(p_i, p_0):
    p_i = numpy.array(p_i)  # Make a copy
    p_0 = numpy.asarray(p_0)

    displacement = p_i - p_0
    radial = numpy.sqrt(displacement[1] ** 2 + displacement[2] ** 2)
    depth = displacement[0]
    return p_i, radial, depth

class CMOS:

    @staticmethod
    def get_noise(shape, rng, dtype=numpy.float64):
        ## get detector noise (photoelectrons)
        noise_data = numpy.loadtxt(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/detector/RNDist_F40.csv'),
            delimiter=',')
        Nr_cmos = noise_data[: , 0]
        p_noise = noise_data[: , 1]
        p_nsum  = p_noise.sum()
        noise = numpy.zeros(shape, dtype=dtype)
        for i in range(shape[0]):
            for j in range(shape[1]):
                ## get detector noise (photoelectrons)
                noise[i, j]  = rng.choice(Nr_cmos, None, p=p_noise / p_nsum)
        return noise

    @staticmethod
    def get_signal(expected, rng=None, processes=None):
        signal = numpy.zeros(expected.shape, dtype=expected.dtype)
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                signal[i, j] = rng.poisson(expected[i, j], None)
        return signal

class EMCCD:

    @staticmethod
    def get_noise(shape, readout_noise, rng, dtype=numpy.float64):
        ## get detector noise (photoelectrons)
        if readout_noise <= 0:
            return numpy.zeros(shape, dtype=dtype)
        return rng.normal(0, readout_noise, shape)

    @staticmethod
    def probability(S, E, a):
        # numpy.sqrt(a * E / S) * numpy.exp(-a * S - E + 2 * numpy.sqrt(a * E * S)) * i1e(2 * numpy.sqrt(a * E * S))
        X = a * S
        Y = 2 * numpy.sqrt(E * X)
        return (1.0 / Y * numpy.exp(-X + Y) * i1e(Y)) # * (2 * a * E * numpy.exp(-E))

    @staticmethod
    def probability_distribution(expected, emgain):
        ## set probability distributions
        sigma = numpy.sqrt(expected) * 5 + 10
        s_min = max(0, emgain * int(expected - sigma))
        s_max = emgain * int(expected + sigma)
        # s = numpy.array([k for k in range(s_min, s_max)])
        S = numpy.arange(s_min, s_max)

        a = 1.0 / emgain
        if S[0] > 0:
            p_signal = EMCCD.probability(S, expected, a)
        else:
            # assert S[1] > 0
            p_signal = numpy.zeros(shape=(len(S), ))
            # p_signal[0] = numpy.exp(-expected)
            p_signal[0] = 1.0 / (2 * a * expected)
            p_signal[1: ] = EMCCD.probability(S[1: ], expected, a)

        p_signal /= p_signal.sum()
        return S, p_signal

    @staticmethod
    def draw_signal(expected, emgain, rng=None):
        if expected <= 0:
            return 0
        s, p_signal = EMCCD.probability_distribution(expected, emgain)
        ## get signal (photoelectrons)
        signal = (rng or numpy.random).choice(s, None, p=p_signal)
        return signal

    @staticmethod
    def get_signal(expected, emgain, rng=None, processes=None):
        if processes is not None and processes > 1:
            with Pool(processes) as pool:
                signal = pool.map(functools.partial(EMCCD.get_signal, rng=None, emgain=emgain), numpy.array_split(expected.flatten(), processes))
            signal = numpy.hstack(signal).reshape(expected.shape)
        else:
            # epected could be one-dimensional. See above.
            shape = expected.shape
            expected = expected.flatten()
            signal = numpy.zeros(expected.size, dtype=expected.dtype)
            for i in range(signal.size):
                signal[i] = EMCCD.draw_signal(expected[i], emgain, rng)
            signal = signal.reshape(shape)
        return signal

class CCD:

    @staticmethod
    def get_noise(shape, readout_noise, rng, dtype=numpy.float64):
        ## get detector noise (photoelectrons)
        if readout_noise <= 0:
            return numpy.zeros(shape, dtype=dtype)
        return rng.normal(0, readout_noise, shape)

    @staticmethod
    def get_signal(expected, rng=None, processes=None):
        signal = numpy.zeros(expected.shape, dtype=expected.dtype)
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                signal[i, j] = rng.poisson(expected[i, j], None)
        return signal

class PhysicalEffectConfigs:
    '''
    Physical effects setting class

        Fluorescence
        Photo-bleaching
        Photo-blinking
    '''

    def __init__(self, config):
        self.initialize(config)

    def initialize(self, config):
        self.fluorescence_bleach = []
        self.fluorescence_budget = []
        self.fluorescence_state  = []

        self.set_background(**config.background)
        self.set_fluorescence(**config.fluorescence)
        self.set_photobleaching(**config.photo_bleaching)
        # self.set_photoactivation(**config.photo_activation)
        # self.set_photoblinking(**config.photo_blinking)

    def set_background(self, mean=None, switch=True):
        self.background_switch = switch
        self.background_mean = mean
        _log.info('--- Background: ')
        _log.info('    Mean = {} photons'.format(self.background_mean))

    def set_fluorescence(self, quantum_yield=None, abs_coefficient=None):
        self.quantum_yield = quantum_yield
        self.abs_coefficient = abs_coefficient
        _log.info('--- Fluorescence: ')
        _log.info('    Quantum Yield =  {}'.format(self.quantum_yield))
        _log.info('    Abs. Coefficient =  {} 1/(M cm)'.format(self.abs_coefficient))
        _log.info('    Abs. Cross-section =  {} cm^2'.format((numpy.log(10) * self.abs_coefficient * 0.1 / constants.N_A) * 1e+4))

    def set_photobleaching(self, half_life=None, switch=True):
        self.photobleaching_switch = switch
        self.photobleaching_half_life = half_life
        _log.info('--- Photobleaching: ')
        _log.info('    Photobleaching half life  =  {}'.format(self.photobleaching_half_life))

    # def set_photoactivation(self, turn_on_ratio=None, activation_yield=None, frac_preactivation=None, switch=True):
    #     self.photoactivation_switch = switch
    #     self.photoactivation_turn_on_ratio = turn_on_ratio
    #     self.photoactivation_activation_yield = activation_yield
    #     self.photoactivation_frac_preactivation = frac_preactivation
    #     _log.info('--- Photoactivation: ')
    #     _log.info('    Turn-on Ratio  =  {}'.format(self.photoactivation_turn_on_ratio))
    #     _log.info('    Effective Ratio  =  {}'.format(self.photoactivation_activation_yield * self.photoactivation_turn_on_ratio / (1 + self.photoactivation_frac_preactivation * self.photoactivation_turn_on_ratio)))
    #     _log.info('    Reaction Yield =  {}'.format(self.photoactivation_activation_yield))
    #     _log.info('    Fraction of Preactivation =  {}'.format(self.photoactivation_frac_preactivation))

    # def set_photoblinking(self, t0_on=None, a_on=None, t0_off=None, a_off=None, switch=True):
    #     self.photoblinking_switch = switch
    #     self.photoblinking_t0_on = t0_on
    #     self.photoblinking_a_on = a_on
    #     self.photoblinking_t0_off = t0_off
    #     self.photoblinking_a_off = a_off
    #     _log.info('--- Photo-blinking: ')
    #     _log.info('    (ON)  t0 =  {} sec'.format(self.photoblinking_t0_on))
    #     _log.info('    (ON)  a  =  {}'.format(self.photoblinking_a_on))
    #     _log.info('    (OFF) t0 =  {} sec'.format(self.photoblinking_t0_off))
    #     _log.info('    (OFF) a  =  {}'.format(self.photoblinking_a_off))

    # def get_prob_bleach(self, tau, dt):
    #     # set the photobleaching-time
    #     tau0  = self.photobleaching_tau0
    #     alpha = self.photobleaching_alpha

    #     # set probability
    #     prob = self.prob_levy(tau, tau0, alpha)
    #     norm = prob.sum()
    #     p_bleach = prob/norm

    #     return p_bleach

    # def get_prob_blink(self, tau_on, tau_off, dt):
    #     # time scale
    #     t0_on  = self.photoblinking_t0_on
    #     a_on   = self.photoblinking_a_on
    #     t0_off = self.photoblinking_t0_off
    #     a_off  = self.photoblinking_a_off

    #     # ON-state
    #     prob_on = self.prob_levy(tau_on, t0_on, a_on)
    #     norm_on = prob_on.sum()
    #     p_blink_on = prob_on/norm_on

    #     # OFF-state
    #     prob_off = self.prob_levy(tau_off, t0_off, a_off)
    #     norm_off = prob_off.sum()
    #     p_blink_off = prob_off/norm_off

    #     return p_blink_on, p_blink_off

    # def get_photobleaching_property(self, dt, n_emit0, rng):
    #     # set the photobleaching-time
    #     tau0  = self.photobleaching_tau0

    #     # set photon budget
    #     photon0 = (tau0/dt)*n_emit0

    #     tau_bleach = numpy.array([j*dt + tau0 for j in range(int(1e+7))])
    #     p_bleach = self.get_prob_bleach(tau_bleach, dt)

    #     # get photon budget and photobleaching-time
    #     tau = rng.choice(tau_bleach, 1, p=p_bleach)
    #     budget = (tau/tau0)*photon0

    #     return tau, budget

    # def set_photophysics_4palm(self, start, end, dt, f, F, N_part, rng):
    #     ##### PALM Configuration
    #     NNN = int(1e+7)

    #     # set the photobleaching-time
    #     tau0  = self.photobleaching_tau0
    #     alpha = self.photobleaching_alpha

    #     # set photon budget
    #     # photon0 = (tau0/dt)*N_emit0
    #     photon0 = (tau0/dt)*1.0

    #     tau_bleach = numpy.array([j*dt+tau0 for j in range(NNN)])
    #     prob = self.prob_levy(tau_bleach, tau0, alpha)
    #     norm = prob.sum()
    #     p_bleach = prob/norm

    #     # set photoblinking (ON/OFF) probability density function (PDF)
    #     if (self.photoblinking_switch == True):

    #         # time scale
    #         t0_on  = self.photoblinking_t0_on
    #         a_on   = self.photoblinking_a_on
    #         t0_off = self.photoblinking_t0_off
    #         a_off  = self.photoblinking_a_off

    #         tau_on  = numpy.array([j*dt+t0_on  for j in range(NNN)])
    #         tau_off = numpy.array([j*dt+t0_off for j in range(NNN)])

    #         # ON-state
    #         prob_on = self.prob_levy(tau_on, t0_on, a_on)
    #         norm_on = prob_on.sum()
    #         p_on = prob_on/norm_on

    #         # OFF-state
    #         prob_off = self.prob_levy(tau_off, t0_off, a_off)
    #         norm_off = prob_off.sum()
    #         p_off = prob_off/norm_off

    #     # sequences
    #     budget = []
    #     state_act = []

    #     # get random number
    #     # numpy.random.seed()

    #     # get photon budget and photobleaching-time
    #     tau = rng.choice(tau_bleach, N_part, p=p_bleach)

    #     for i in range(N_part):

    #         # get photon budget and photobleaching-time
    #         photons = (tau[i]/tau0)*photon0

    #         N = int((end - start)/dt)

    #         time  = numpy.array([j*dt + start for j in range(N)])
    #         state = numpy.zeros(shape=(N))

    #         #### Photoactivation: overall activation yeild
    #         p = self.photoactivation_activation_yield
    #         # q = self.photoactivation_frac_preactivation

    #         r = rng.uniform(0, 1, N/F*f)
    #         s = (r < p).astype('int')

    #         index = numpy.abs(s - 1).argmin()
    #         t0 = (index/f*F + numpy.remainder(index, f))*dt + start

    #         t1 = t0 + tau[i]

    #         N0 = (numpy.abs(time - t0)).argmin()
    #         N1 = (numpy.abs(time - t1)).argmin()
    #         state[N0:N1] = numpy.ones(shape=(N1-N0))

    #         if (self.photoblinking_switch == True):

    #             # ON-state
    #             t_on  = rng.choice(tau_on,  100, p=p_on)
    #             # OFF-state
    #             t_off = rng.choice(tau_off, 100, p=p_off)

    #             k = (numpy.abs(numpy.cumsum(t_on) - tau[i])).argmin()

    #             # merge t_on/off arrays
    #             t_blink = numpy.array([t_on[0:k], t_off[0:k]])
    #             t = numpy.reshape(t_blink, 2*k, order='F')

    #             i_on  = N0

    #             for j in range(k):

    #                 f_on  = i_on  + int(t[j]/dt)
    #                 i_off = f_on
    #                 f_off = i_off + int(t[j+1]/dt)

    #                 state[i_on:f_on]   = 1
    #                 state[i_off:f_off] = 0

    #                 i_on = f_off

    #         # set fluorescence state (without photoblinking)
    #         budget.append(photons)
    #         state_act.append(state)

    #     self.fluorescence_bleach = numpy.array(tau)
    #     self.fluorescence_budget = numpy.array(budget)
    #     self.fluorescence_state  = numpy.array(state_act)

    # def set_photophysics_4lscm(self, start, end, dt, N_part, rng):
    #     ##### LSCM Configuration

    #     NNN = int(1e+7)

    #     # set the photobleaching-time
    #     tau0  = self.photobleaching_tau0
    #     alpha = self.photobleaching_alpha

    #     # set photon budget
    #     photon0 = (tau0/dt)*1.0

    #     tau_bleach = numpy.array([j*dt+tau0 for j in range(NNN)])
    #     prob = self.prob_levy(tau_bleach, tau0, alpha)
    #     norm = prob.sum()
    #     p_bleach = prob/norm

    #     # set photoblinking (ON/OFF) probability density function (PDF)
    #     if (self.photoblinking_switch == True):

    #         # time scale
    #         t0_on  = self.photoblinking_t0_on
    #         a_on   = self.photoblinking_a_on
    #         t0_off = self.photoblinking_t0_off
    #         a_off  = self.photoblinking_a_off

    #         tau_on  = numpy.array([j*dt+t0_on  for j in range(NNN)])
    #         tau_off = numpy.array([j*dt+t0_off for j in range(NNN)])

    #         # ON-state
    #         prob_on = self.prob_levy(tau_on, t0_on, a_on)
    #         norm_on = prob_on.sum()
    #         p_on = prob_on/norm_on

    #         # OFF-state
    #         prob_off = self.prob_levy(tau_off, t0_off, a_off)
    #         norm_off = prob_off.sum()
    #         p_off = prob_off/norm_off

    #     # sequences
    #     budget = []
    #     state_act = []

    #     # get random number
    #     # numpy.random.seed()

    #     # get photon budget and photobleaching-time
    #     tau = rng.choice(tau_bleach, N_part, p=p_bleach)

    #     for i in range(N_part):

    #         # get photon budget and photobleaching-time
    #         photons = (tau[i]/tau0)*photon0

    #         N = 10000 #int((end - start)/dt)

    #         time  = numpy.array([j*dt + start for j in range(N)])
    #         state = numpy.zeros(shape=(N))

    #         t0 = start
    #         t1 = t0 + tau[i]

    #         N0 = (numpy.abs(time - t0)).argmin()
    #         N1 = (numpy.abs(time - t1)).argmin()
    #         state[N0:N1] = numpy.ones(shape=(N1-N0))

    #         if (self.photoblinking_switch == True):

    #             # ON-state
    #             t_on  = rng.choice(tau_on,  100, p=p_on)
    #             # OFF-state
    #             t_off = rng.choice(tau_off, 100, p=p_off)

    #             k = (numpy.abs(numpy.cumsum(t_on) - tau[i])).argmin()

    #             # merge t_on/off arrays
    #             t_blink = numpy.array([t_on[0:k], t_off[0:k]])
    #             t = numpy.reshape(t_blink, 2*k, order='F')

    #             i_on  = N0

    #             for j in range(k):

    #                 f_on  = i_on  + int(t[j]/dt)
    #                 i_off = f_on
    #                 f_off = i_off + int(t[j+1]/dt)

    #                 state[i_on:f_on]   = 1
    #                 state[i_off:f_off] = 0

    #                 i_on = f_off

    #         # set photon budget and fluorescence state
    #         budget.append(photons)
    #         state_act.append(state)

    #     self.fluorescence_bleach = numpy.array(tau)
    #     self.fluorescence_budget = numpy.array(budget)
    #     self.fluorescence_state  = numpy.array(state_act)

class EPIFMConfigs:

    def __init__(self, config, rng=None):
        self.initialize(config, rng=rng)

    def initialize(self, config, rng=None):
        """Initialize based on the given config.

        Args:
            config (Configuration): A config object.
            rng (numpy.RandomState, optional): A random number generator
                for initializing this class. Defaults to `None`.

        """
        if config.type.lower() != 'epifm':
            raise ValueError("An invalid type [{}] was given. 'epifm' is required.".format(config.type))

        if rng is None:
            warnings.warn('A random number generator [rng] is not given.')
            rng = numpy.random.RandomState()

        self.set_fluorophore(**config.fluorophore)  # => self.__wave_length

        self.set_shutter(**config.shutter)
        self.set_light_source(**config.light_source)
        self.set_dichroic_mirror(**config.dichroic_mirror)

        self.image_magnification = config.magnification
        _log.info('--- Magnification: x {}'.format(self.image_magnification))

        self.set_detector(**config.detector)
        self.set_analog_to_digital_converter(rng=rng, **config.analog_to_digital_converter)
        self.set_excitation_filter(**config.excitation_filter)
        self.set_emission_filter(**config.emission_filter)

        # self.fluorophore_psf = self.get_PSF_detector()
        self.fluorophore_psf = PointSpreadingFunction(
            config.fluorophore.radial_cutoff, self.psf_radial_width, config.fluorophore.depth_cutoff,
            self.fluorophore_type, self.psf_wavelength)

        self.effects = PhysicalEffectConfigs(config.effects)

    def set_shutter(self, start_time=None, end_time=None, time_open=None, time_lapse=None, switch=True):
        self.shutter_switch = switch
        self.shutter_start_time = start_time
        self.shutter_end_time = end_time

        _log.info('--- Shutter:')
        _log.info('    Start-Time = {} sec'.format(self.shutter_start_time))
        _log.info('    End-Time   = {} sec'.format(self.shutter_end_time))

    def set_light_source(self, type=None, wave_length=None, flux_density=None, radius=None, angle=None, switch=True):
        self.source_switch = switch
        self.source_type = type
        self.source_wavelength = wave_length
        self.source_flux_density = flux_density
        self.source_radius = radius
        self.source_angle = angle

        _log.info('--- Light Source:{}'.format(self.source_type))
        _log.info('    Wave Length = {} m'.format(self.source_wavelength))
        _log.info('    Beam Flux Density = {} W/m2'.format(self.source_flux_density))
        _log.info('    1/e2 Radius = {} m'.format(self.source_radius))
        _log.info('    Angle = {} radian'.format(self.source_angle))

    def set_fluorophore(
            self, type=None, wave_length=None, normalization=None, radius=None, radial_width=None,
            min_wave_length=None, max_wave_length=None, radial_cutoff=None, depth_cutoff=None):
        self.__wave_length = numpy.arange(min_wave_length, max_wave_length, 1e-9, dtype=float)

        self.fluorophore_type = type
        self.fluorophore_radius = radius
        self.psf_normalization = normalization

        if type == 'Gaussian':
            N = len(self.__wave_length)
            fluorophore_excitation = numpy.zeros(N, dtype=float)
            fluorophore_emission = numpy.zeros(N, dtype=float)
            idx = (numpy.abs(self.__wave_length - wave_length)).argmin()
            index_ex = index_em = idx
            self.psf_radial_width = radial_width
        else:
            (fluorophore_excitation, fluorophore_emission) = io.read_fluorophore_catalog(type)
            fluorophore_excitation = self.calculate_efficiency(fluorophore_excitation, self.__wave_length)
            fluorophore_emission = self.calculate_efficiency(fluorophore_emission, self.__wave_length)
            fluorophore_excitation = numpy.array(fluorophore_excitation)
            fluorophore_emission = numpy.array(fluorophore_emission)
            index_ex = fluorophore_excitation.argmax()
            index_em = fluorophore_emission.argmax()
            if wave_length is not None:
                warnings.warn('The given wave length [{}] was ignored'.format(wave_length))
            self.psf_radial_width = None  #XXX: This is only required for Gaussian

        fluorophore_excitation[index_ex] = 100
        fluorophore_emission[index_em] = 100
        self.fluoex_eff = fluorophore_excitation
        self.fluoem_eff = fluorophore_emission
        fluorophore_excitation /= sum(fluorophore_excitation)
        fluorophore_emission /= sum(fluorophore_emission)
        self.fluoex_norm = fluorophore_excitation
        self.fluoem_norm = fluorophore_emission
        self.psf_wavelength = self.__wave_length[index_em]

        _log.info('--- Fluorophore: {} PSF'.format(self.fluorophore_type))
        _log.info('    Wave Length   =  {} m'.format(self.psf_wavelength))
        _log.info('    Normalization =  {}'.format(self.psf_normalization))
        _log.info('    Fluorophore radius =  {} m'.format(self.fluorophore_radius))
        if self.psf_radial_width is not None:
            _log.info('    Lateral Width =  {} m'.format(self.psf_radial_width))
            # _log.info('    Lateral Width =  {} m'.format(self.psf_width[0]))
            # _log.info('    Axial Width =  {} m'.format(self.psf_width[1]))
        _log.info('    PSF Normalization Factor =  {}'.format(self.psf_normalization))
        _log.info('    Emission  : Wave Length =  {} m'.format(self.psf_wavelength))

    def set_dichroic_mirror(self, type=None, switch=True):
        self.dichroic_switch = switch
        if type is not None:
            dichroic_mirror = io.read_dichroic_catalog(type)
            self.dichroic_eff = self.calculate_efficiency(dichroic_mirror, self.__wave_length)
        else:
            self.dichroic_eff = numpy.zeros(len(self.__wave_length), dtype=float)
        _log.info('--- Dichroic Mirror:')

    def set_detector(
            self, type=None, image_size=None, pixel_length=None, exposure_time=None, focal_point=None,
            QE=None, readout_noise=None, dark_count=None, emgain=None, switch=True):
        self.detector_switch = switch
        self.detector_type = type
        self.detector_image_size = image_size
        self.detector_pixel_length = pixel_length
        self.detector_exposure_time = exposure_time
        self.detector_focal_point = focal_point
        self.detector_qeff = QE
        self.detector_readout_noise = readout_noise
        self.detector_dark_count = dark_count
        self.detector_emgain = emgain
        # self.detector_base_position = base_position
        self.detector_focal_point = focal_point

        _log.info('--- Detector:  {}'.format(self.detector_type))
        if self.detector_image_size is not None:
            _log.info('    Image Size  =  {} x {}'.format(self.detector_image_size[0], self.detector_image_size[1]))
        _log.info('    Pixel Size  =  {} m/pixel'.format(self.detector_pixel_length))
        _log.info('    Focal Point =  {}'.format(self.detector_focal_point))
        _log.info('    Exposure Time =  {} sec'.format(self.detector_exposure_time))
        if self.detector_qeff is not None:
            _log.info('    Quantum Efficiency =  {} %'.format(100 * self.detector_qeff))
        _log.info('    Readout Noise =  {} electron'.format(self.detector_readout_noise))
        _log.info('    Dark Count =  {} electron/sec'.format(self.detector_dark_count))
        _log.info('    EM gain = x {}'.format(self.detector_emgain))
        # _log.info('    Position    =  {}'.format(self.detector_base_position))
        _log.info('Focal Center: {}'.format(self.detector_focal_point))

    def set_analog_to_digital_converter(self, *, rng, bit=None, offset=None, fullwell=None, type=None, count=None):
        self.ADConverter_bit = bit
        self.ADConverter_fullwell = fullwell
        self.ADConverter_fpn_type = type
        self.ADConverter_fpn_count = count

        offset, gain = self.calculate_analog_to_digital_converter_gain(offset, rng)
        self.ADConverter_offset = offset
        self.ADConverter_gain = gain

        _log.info('--- A/D Converter: %d-bit' % (self.ADConverter_bit))
        # _log.info('    Gain = %.3f electron/count' % (self.ADConverter_gain))
        # _log.info('    Offset =  {} count'.format(self.ADConverter_offset))
        _log.info('    Fullwell =  {} electron'.format(self.ADConverter_fullwell))
        _log.info('    {}-Fixed Pattern Noise: {} count'.format(self.ADConverter_fpn_type, self.ADConverter_fpn_count))

    def calculate_analog_to_digital_converter_gain(self, ADC0, rng):
        # image pixel-size
        Nw_pixel = self.detector_image_size[0]
        Nh_pixel = self.detector_image_size[1]

        # set ADC parameters
        bit  = self.ADConverter_bit
        fullwell = self.ADConverter_fullwell
        # ADC0 = self.ADConverter_offset

        # set Fixed-Pattern noise
        FPN_type = self.ADConverter_fpn_type
        FPN_count = self.ADConverter_fpn_count

        # if FPN_type is None:
        if FPN_type == 'none':
            offset = numpy.full(Nw_pixel * Nh_pixel, ADC0)
        elif FPN_type == 'pixel':
            if rng is None:
                raise RuntimeError('A random number generator is required.')
            offset = numpy.rint(rng.normal(ADC0, FPN_count, Nw_pixel * Nh_pixel))
        elif FPN_type == 'column':
            if rng is None:
                raise RuntimeError('A random number generator is required.')
            column = rng.normal(ADC0, FPN_count, Nh_pixel)
            temporal = numpy.tile(column, (1, Nw_pixel))
            offset = numpy.rint(temporal.reshape(Nh_pixel * Nw_pixel))
        else:
            raise ValueError("FPN type [{}] is invalid ['pixel', 'column' or 'none']".format(FPN_type))

        # set ADC gain
        # gain = numpy.array(map(lambda x: (fullwell - 0.0) / (pow(2.0, bit) - x), offset))
        gain = (fullwell - 0.0) / (pow(2.0, bit) - offset)

        offset = offset.reshape([Nw_pixel, Nh_pixel])
        gain = gain.reshape([Nw_pixel, Nh_pixel])

        return (offset, gain)

    def set_excitation_filter(self, type=None, switch=True):
        self.excitation_switch = switch
        if type is not None:
            excitation_filter = io.read_excitation_catalog(type)
            self.excitation_eff = self.calculate_efficiency(excitation_filter, self.__wave_length)
        else:
            self.excitation_eff = numpy.zeros(len(self.__wave_length), dtype=float)
        _log.info('--- Excitation Filter:')

    def set_emission_filter(self, type=None, switch=True):
        self.emission_switch = switch
        if type is not None:
            emission_filter = io.read_emission_catalog(type)
            self.emission_eff = self.calculate_efficiency(emission_filter, self.__wave_length)
        else:
            self.emission_eff = numpy.zeros(len(self.__wave_length), dtype=float)
        _log.info('--- Emission Filter:')

    @staticmethod
    def calculate_efficiency(data, wave_length):
        data = numpy.array(data, dtype = 'float')
        data = data[data[: , 0] % 1 == 0, :]

        efficiency = numpy.zeros(len(wave_length))

        wave_length = numpy.round(wave_length / 1e-9).astype(int)  #XXX: numpy.array(dtype=int)
        idx1 = numpy.in1d(wave_length, data[:, 0])
        idx2 = numpy.in1d(numpy.array(data[:, 0]), wave_length)

        efficiency[idx1] = data[idx2, 1]

        return efficiency.tolist()

class _EPIFMSimulator:
    '''
    A class of the simulator for Epifluorescence microscopy (EPI).
    '''

    def __init__(self, configs, environ=None):
        """A constructor of _EPIFMSimulator.

        Args:
            config (Config, optional): A config object for initializing this class.
            rng (numpy.RandomState, optional): A random number generator
                for initializing this class.

        """
        self.configs = configs

        if environ is not None:
            self.environ = environ

    def generate_frames(
            self, input_data, num_frames, start_time=0.0, exposure_time=None,
            rng=None, processes=None):
        """Output all images from the given particle data.

        Args:
            input_data (list): An input data. A list of pairs of time and a list of particles.
                Each particle is represented as a list of numbers: a coordinate (a triplet of floats),
                molecule id, and a state of fluorecent.
                The number of particles in each frame must be static.
            num_frames (int): The number of frames taken.
            start_time (float, optional): A time to start detecting.
                Defaults to 0.0.
            exposure_time (float, optional): An exposure time.
                Defaults to `detector_exposure_time` in the configuration.
            rng (numpy.RandomState, optional): A random number generator.

        """
        if rng is None:
            _log.info('A random number generator was initialized.')
            rng = numpy.random.RandomState()

        fluorescence_states = None
        if self.configs.effects.photobleaching_switch:
            fluorescence_states = {}

        exposure_time = exposure_time or self.configs.detector_exposure_time

        for frame_index in range(num_frames):
            camera, infodict = self.output_frame(
                    input_data, frame_index=frame_index, start_time=start_time, exposure_time=exposure_time,
                    fluorescence_states=fluorescence_states, rng=rng, processes=processes)
            yield (camera, infodict)

    def output_frames(
            self, input_data, num_frames, start_time=0.0, exposure_time=None,
            pathto='./images', data_fmt='image_%07d.npy', true_fmt='true_%07d.npy',
            image_fmt='image_%07d.png', cmin=None, cmax=None, low=None, high=None,
            rng=None, processes=None):
        """Output all images from the given particle data.

        Args:
            input_data (list): An input data. A list of pairs of time and a list of particles.
                Each particle is represented as a list of numbers: a coordinate (a triplet of floats),
                molecule id, and a state of fluorecent.
                The number of particles in each frame must be static.
            num_frames (int): The number of frames taken.
            start_time (float, optional): A time to start detecting.
                Defaults to 0.0.
            exposure_time (float, optional): An exposure time.
                Defaults to `detector_exposure_time` in the configuration.
            pathto (str, optional): A path to save images and ndarrays. Defaults to './images'.
            image_fmt (str, optional): A format of the filename to save 8-bit images.
                An int is available as its frame index. Defaults to 'image_%07d.png'.
                If None, no file is saved.
            data_fmt (str, optional): A format of the filename to save the detected photon counts.
                An int is available as its frame index. Defaults to 'image_%07d.npy'.
                If None, no file is saved.
            true_fmt (str, optional): A format of the filename to save true data.
                An int is available as its frame index. Defaults to 'true_%07d.npy'.
                If None, no file is saved.
            cmin (int, optional): A minimal value used to generate 8-bit images.
            cmax (int, optional): A maximum value used to generate 8-bit images.
            low (int, optional): A minimal value used to generate 8-bit images.
                Defaults to 0.
            high (int, optional): A maximum value used to generate 8-bit images.
                Defaults to 255.
            rng (numpy.RandomState, optional): A random number generator.

        """
        # Check and create the folders for image and output files.
        if not os.path.exists(pathto):
            os.makedirs(pathto)

        low = 0 if low is None else low
        high = 255 if high is None else high

        if rng is None:
            _log.info('A random number generator was initialized.')
            rng = numpy.random.RandomState()

        results = []
        for camera, infodict in self.generate_frames(
                input_data, frame_index=frame_index, start_time=start_time, exposure_time=exposure_time,
                fluorescence_states=fluorescence_states, rng=rng, processes=processes):
            # save photon counts to numpy-binary file
            if data_fmt is not None:
                data_file_name = os.path.join(pathto, data_fmt % (frame_index))
                numpy.save(data_file_name, camera)

            # save true-dataset to numpy-binary file
            if true_fmt is not None:
                raise RuntimeError("Not supported.")
                # true_file_name = os.path.join(pathto, true_fmt % (frame_index))
                # numpy.save(true_file_name, true_data)

            # save images to numpy-binary file
            if image_fmt is not None:
                image_file_name = os.path.join(pathto, image_fmt % (frame_index))
                Image(camera[: , : , 1]).as_8bit(cmin, cmax, low, high).save(image_file_name)

            results.append((camera, infodict))
        return results

    def output_frame(
            self, input_data, frame_index=0, start_time=0.0, exposure_time=None,
            fluorescence_states=None, rng=None, processes=None):
        """Output an image from the given particle data.

        Args:
            input_data (list): An input data. A list of pairs of time and a list of particles.
                Each particle is represented as a list of numbers: a coordinate (x, y, z),
                molecule id, and a state of fluorecent.
                The number of particles in each frame must be static.
            frame_index (int, optional): An index of the frame. The value must be 0 and more.
                Defaults to 0.
            start_time (float, optional): A time to start detecting.
                Defaults to 0.0.
            exposure_time (float, optional): An exposure time.
                Defaults to `detector_exposure_time` in the configuration.
            rng (numpy.RandomState, optional): A random number generator.

        """
        processes = processes or (1 if self.environ is None else self.environ.processes)
        exposure_time = exposure_time or self.configs.detector_exposure_time

        if rng is None:
            _log.info('A random number generator was initialized.')
            rng = numpy.random.RandomState()

        times = numpy.array([t for t, _ in input_data])
        t = start_time + exposure_time * frame_index

        if self.configs.shutter_switch:
            t = max(t, self.configs.shutter_start_time)
            exposure_time = max(0.0, min(t + exposure_time, self.configs.shutter_end_time))

        start_index = numpy.searchsorted(times, t, side='right')
        if start_index != 0:
            start_index -= 1
        stop_index = numpy.searchsorted(times, t + exposure_time, side='left')
        if times[start_index] > t:
            warnings.warn("No data input for interval [{}, {}]".format(t, times[start_index]))

        frame_data = input_data[start_index: stop_index]

        _log.info('time: {} sec ({})'.format(t, frame_index))

        # focal point
        p_0 = numpy.asarray(self.configs.detector_focal_point)  # meter-scale
        _log.info('center (width, height): {} {}'.format(p_0[0], p_0[1]))

        # beam position: Assuming beam position = focal point (for temporary)
        p_b = copy.copy(p_0)

        # camera pixels
        Nw_pixel, Nh_pixel = self.configs.detector_image_size  # pixels
        pixel_length = self.configs.detector_pixel_length / self.configs.image_magnification  # m-scale
        camera_pixel = numpy.zeros((Nw_pixel, Nh_pixel, 2))
        _log.info('scaling [m/pixel]: {}'.format(pixel_length))

        optinfo = {}

        # loop for frame data
        for i, (i_time, particles) in enumerate(frame_data):
            # When the first i_time is less than the start time given, exposure_time is shorter than expected
            current_time = i_time if i != 0 else t
            next_time = frame_data[i + 1][0] if i + 1 < len(frame_data) else t + exposure_time
            unit_time = next_time - current_time

            _log.info('    {:03d}-th file in {:03d}-th frame: {} + {} sec'.format(
                i, frame_index, current_time, unit_time))

            if unit_time < 1e-13:  #XXX: Skip merging when the exposure time is too short
                continue

            # loop for particles
            expected, optinfo_, states = self.get_molecule_plane(
                    particles, shape=camera_pixel.shape[: 2], p_b=p_b, p_0=p_0, unit_time=unit_time,
                    optional_info=optinfo, fluorescence_states=fluorescence_states, rng=rng, processes=processes)
            camera_pixel[:, :, 0] += expected
            optinfo.update(optinfo_)
            if fluorescence_states is not None:
                fluorescence_states.update(states)

        for m_id in optinfo:
            optinfo[m_id][1] /= exposure_time  # photon state
            optinfo[m_id][2: 6] /= optinfo[m_id][0]  # Calculating time averages

            # Transforming CoM (X and Y) from coordinates to pixels.
            # The coordinate is positioned at the center of a pixel, rather than the corner.
            # Therefore, shifting not `Nw_pixel * 0.5`, but `(Nw_pixel - 1) * 0.5`.
            # See also `PointSpreadingFunction.overlay_signal`_
            optinfo[m_id][2] = (optinfo[m_id][2] - p_0[1]) / pixel_length + (Nw_pixel - 1) * 0.5
            optinfo[m_id][3] = (optinfo[m_id][3] - p_0[2]) / pixel_length + (Nh_pixel - 1) * 0.5

        # apply detector effects
        camera = self.__detector_output(rng, camera_pixel, processes=processes)
        # return (camera, optinfo)
        infodict = dict(true_data=optinfo)
        if fluorescence_states is not None:
            infodict['fluorescence_states'] = copy.copy(fluorescence_states)
        return (camera, infodict)

    def get_molecule_plane(
            self, particles, shape, p_b, p_0, unit_time,
            optional_info, fluorescence_states, rng=None, processes=None):
        expected = numpy.zeros(shape)
        optinfo = {int(m_id): optional_info[int(m_id)]
                for _, _, _, m_id, _ in particles if int(m_id) in optional_info}
        if fluorescence_states is None:
            states = None
        else:
            states = {int(m_id): fluorescence_states[int(m_id)]
                    for _, _, _, m_id, _ in particles if int(m_id) in fluorescence_states}

        if processes is not None and processes > 1:
            if fluorescence_states is not None and self.configs.effects.photobleaching_switch:
                amplitude0, _ = self.snells_law()
                N_emit0 = self.__get_emit_photons(amplitude0, 1.0)
                half_life = self.configs.effects.photobleaching_half_life
                for p in particles:
                    m_id = int(p[3])
                    if m_id not in states:
                        states[m_id] = self.get_photon_budget(
                                N_emit0, half_life, rng=rng)

            func = functools.partial(
                    self.get_molecule_plane,
                    shape=shape, p_b=p_b, p_0=p_0, unit_time=unit_time,
                    optional_info=optinfo, fluorescence_states=states, rng=None, processes=None)
            with Pool(processes) as pool:
                results = pool.map(func, numpy.array_split(particles, processes))
            for expected_, optinfo_, states_ in results:
                expected += expected_
                optinfo.update(optinfo_)
                if states is not None:
                    states.update(states_)
        else:
            for i in range(len(particles)):
                self.__overlay_molecule_plane(
                        expected, particles[i], p_b, p_0, unit_time, optinfo, states, rng)
        return expected, optinfo, states

    def __overlay_molecule_plane(
            self, expected, particle_i, _p_b, p_0, unit_time,
            optional_info, fluorescence_states=None, rng=None):
        # m-scale
        # p_b (ndarray): beam position (assumed to be the same with focal center), but not used.
        # p_0 (ndarray): focal center

        # particles coordinate, molecule id, photon state
        x, y, z, m_id, p_state = particle_i
        m_id = int(m_id)

        p_i = numpy.array([x, y, z])

        # Snell's law
        amplitude, penet_depth = self.snells_law()  #XXX: This might depends on p_b and p_i

        # particles coordinate in real(nm) scale
        _, radial, depth = cylindrical_coordinate(p_i, p_0)

        # get exponential amplitude (only for TIRFM-configuration)
        # depth must be positve in case of TIRFM
        #XXX: amplitude = amplitude * numpy.exp(-depth / penet_depth)
        amplitude = amplitude * numpy.exp(-abs(depth) / penet_depth)

        # get the number of photons emitted
        N_emit = self.__get_emit_photons(amplitude, unit_time)

        if fluorescence_states is not None:
            if self.configs.effects.photobleaching_switch:
                if m_id not in fluorescence_states:
                    assert rng is not None
                    amplitude0, _ = self.snells_law()
                    N_emit0 = self.__get_emit_photons(amplitude0, 1.0)
                    fluorescence_states[m_id] = self.get_photon_budget(
                            N_emit0, self.configs.effects.photobleaching_half_life, rng=rng)  #XXX: rng is only required here
                budget = fluorescence_states[m_id] - N_emit
                if budget <= 0:
                    budget = 0
                    p_state = 0.0  #XXX
                fluorescence_states[m_id] = budget

        # Photon Transmission Efficiency
        normalization = self.configs.fluoem_norm
        if self.configs.dichroic_switch:
            normalization *= 0.01 * self.configs.dichroic_eff
        if self.configs.emission_switch:
            normalization *= 0.01 * self.configs.emission_eff
        normalization = numpy.sum(normalization) * self.configs.psf_normalization
        normalization *= p_state * N_emit / (4.0 * numpy.pi)

        # add signal matrix to image plane
        pixel_length = self.configs.detector_pixel_length / self.configs.image_magnification
        self.configs.fluorophore_psf.overlay_signal(expected, p_i - p_0, pixel_length, normalization)

        # set true-dataset
        if m_id not in optional_info:
            optional_info[m_id] = numpy.zeros(8, dtype=numpy.float64)
        optional_info[m_id] += numpy.array([
            unit_time,
            unit_time * p_state,  # photon-state
            unit_time * p_i[1],  # X-coordinate in the image-plane
            unit_time * p_i[2],  # Y-coordinate in the image-plane
            unit_time * p_i[1],  # X-coordinate in the image-plane
            unit_time * p_i[2],  # Y-coordinate in the image-plane
            unit_time * depth,  # Depth from focal-plane
            normalization,  # normalization
            ])

    def __get_emit_photons(self, amplitude, unit_time):
        # Absorption coeff [1/(cm M)]
        abs_coeff = self.configs.effects.abs_coefficient
        # Quantum yield
        QY = self.configs.effects.quantum_yield
        fluorophore_radius = self.configs.fluorophore_radius
        return self.get_emit_photons(amplitude, unit_time, abs_coeff, QY, fluorophore_radius)

    @staticmethod
    def get_emit_photons(amplitude, unit_time, abs_coeff, QY, fluorophore_radius):
        # Cross-section [m2]
        x_sec = numpy.log(10) * abs_coeff * 0.1 / constants.N_A

        # get the number of absorption photons: [#/(m2 sec)]*[m2]*[sec]
        n_abs = amplitude * x_sec * unit_time

        # Beer-Lamberts law: A = log(I0 / I) = abs coef. * concentration * path length ([m2]*[#/m3]*[m])
        fluorophore_volume = (4.0 / 3.0) * numpy.pi * numpy.power(fluorophore_radius, 3)
        fluorophore_depth  = 2.0 * fluorophore_radius

        A = (abs_coeff * 0.1 / constants.N_A) * (1.0 / fluorophore_volume) * fluorophore_depth

        # get the number of photons emitted
        n_emit = QY * n_abs * (1.0 - numpy.power(10.0, -A))

        return n_emit

    def snells_law(self):
        """Snell's law.

        Returns:
            float: Amplitude
            float: Penetration depth

        """
        # x_0, y_0, z_0 = p_0
        # x_i, y_i, z_i = p_i

        # Illumination: Assume that uniform illumination (No gaussian)
        # flux density, [W / m ** 2] = [J / s / m ** 2] = [kg / s ** 3]
        P_0 = self.configs.source_flux_density

        # single photon energy
        wave_length = self.configs.source_wavelength  # m
        E_wl = constants.hc / wave_length

        # photon flux density [photons/sec/m2]
        N_0 = P_0 / E_wl

        # Incident beam: Amplitude
        A2_Is = N_0
        A2_Ip = N_0

        # incident beam angle (radian)
        angle = self.configs.source_angle
        # theta_in = (angle / 180.) * numpy.pi
        theta_in = angle

        sin_th1 = numpy.sin(theta_in)
        cos_th1 = numpy.cos(theta_in)

        sin = sin_th1
        cos = cos_th1
        sin2 = sin ** 2
        cos2 = cos ** 2

        # index of refraction
        n_1 = 1.46  # fused silica
        n_2 = 1.384 # cell
        # n_3 = 1.337 # culture medium

        r  = n_2 / n_1
        r2 = r ** 2

        # Epi-illumination at apical surface
        if sin2 / r2 < 1:
            amplitude = N_0  # = (A2_Ip + A2_Is) / 2
            penetration_depth = numpy.inf
        else:
            # TIRF-illumination at basal cell-surface
            # Evanescent field: Amplitude and Depth
            # Assume that the s-polar direction is parallel to y-axis
            A2_x = A2_Ip * (4 * cos2 * (sin2 - r2) / (r2 ** 2 *cos2 + sin2 - r2))
            A2_y = A2_Is * (4 * cos2 / (1 - r2))
            A2_z = A2_Ip * (4 * cos2 * sin2 / (r2 ** 2 * cos2 + sin2 - r2))

            # set illumination amplitude
            A2_Tp = A2_x + A2_z
            A2_Ts = A2_y
            amplitude = (A2_Tp + A2_Ts) / 2

            penetration_depth = wave_length / (4.0 * numpy.pi * numpy.sqrt(n_1 ** 2 * sin2 - n_2 ** 2))

        return amplitude, penetration_depth

    def __detector_output(self, rng, camera_pixel, processes=1):
        ## Detector: Quantum Efficiency
        # index = int(self.configs.psf_wavelength / 1e-9) - int(self.configs.wave_length[0] / 1e-9)
        # QE = self.configs.detector_qeff[index]
        QE = self.configs.detector_qeff
        ## get signal (photons)
        photons = camera_pixel[:, :, 0]
        ## get constant background (photoelectrons)
        if self.configs.effects.background_switch:
            photons += self.configs.effects.background_mean
        ## get signal (expectation)
        expected = QE * photons

        ## conversion: photon --> photoelectron --> ADC count
        ## select Camera type
        if self.configs.detector_type == "CMOS":
            noise = CMOS.get_noise(expected.shape, rng=rng)
            signal = CMOS.get_signal(expected, rng=rng, processes=processes)
        elif self.configs.detector_type == "EMCCD":
            noise = EMCCD.get_noise(expected.shape, readout_noise=self.configs.detector_readout_noise, rng=rng)
            signal = EMCCD.get_signal(
                    expected,
                    emgain=self.configs.detector_emgain,
                    rng=rng,
                    processes=processes)
        elif self.configs.detector_type == "CCD":
            noise = CCD.get_noise(expected.shape, readout_noise=self.configs.detector_readout_noise, rng=rng)
            signal = CCD.get_signal(expected, rng=rng, processes=processes)
        else:
            raise RuntimeError(
                    "Unknown detector type was given [{}]. ".format(self.configs.detector_type)
                    + "Use either one of 'CMOS', 'CCD' or 'EMCCD'.")

        camera_pixel[:, :, 0] = expected
        camera_pixel[:, :, 1] = self.__get_analog_to_digital_converter_counts(
                signal + noise,
                fullwell=self.configs.ADConverter_fullwell,
                gain=self.configs.ADConverter_gain,
                offset=self.configs.ADConverter_offset,
                bit=self.configs.ADConverter_bit)
        return camera_pixel

    @staticmethod
    def __get_analog_to_digital_converter_counts(photoelectron, fullwell, gain, offset, bit):
        # check non-linearity
        ADC = photoelectron.copy()
        ADC[ADC > fullwell] = fullwell

        # convert photoelectron to ADC counts
        ADC_max = 2 ** bit - 1
        ADC /= gain
        ADC += offset
        ADC[ADC > ADC_max] = ADC_max
        ADC[ADC < 0] = 0
        return ADC

    @staticmethod
    def get_photon_budget(N_emit0, half_life, rng):
        beta = half_life / numpy.log(2.0)
        bleaching_time = rng.exponential(scale=beta, size=None)
        budget = bleaching_time * N_emit0
        return budget
