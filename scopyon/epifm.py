import copy
import math
import warnings
import os

from multiprocessing import Pool
import functools

import numpy

import scipy.special

from scipy.special import j0, i1e
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from .effects import PhysicalEffects
from . import io
from .config import Config
from .image import save_image, convert_8bit
from . import constants

from logging import getLogger
_log = getLogger(__name__)


# def levy_probability_function(t, t0, a):
#     return (a / t0) * numpy.power(t0 / t, 1 + a)
#     # return numpy.power(t0 / t, 1 + a)

class PointSpreadingFunction:

    def __init__(self, psf_radial_cutoff, psf_width, psf_depth_cutoff, fluoem_norm, dichroic_switch, dichroic_eff, emission_switch, emission_eff, fluorophore_type, psf_wavelength, psf_normalization):
        # fluorophore
        self.fluoem_norm = fluoem_norm
        self.fluorophore_type = fluorophore_type
        self.psf_wavelength = psf_wavelength
        self.psf_normalization = psf_normalization
        self.psf_radial_cutoff = psf_radial_cutoff
        self.psf_width = psf_width
        self.psf_depth_cutoff = psf_depth_cutoff

        # else
        self.dichroic_switch = dichroic_switch
        self.dichroic_eff = dichroic_eff
        self.emission_switch = emission_switch
        self.emission_eff = emission_eff

        self.__normalization = self.__get_normalization()
        self.__fluorophore_psf = {}
        self.__resolution_depth = 1e-9
        self.__resolution_radial = 1e-9

    def get(self, depth):
        depth = abs(depth)
        if depth < self.psf_depth_cutoff + self.__resolution_depth:
            assert depth >= 0
            key = int(depth / self.__resolution_depth)
            depth = key * self.__resolution_depth
        else:
            key = -1
            depth = self.psf_depth_cutoff

        if key not in self.__fluorophore_psf:
            self.__fluorophore_psf[key] = self.__get(depth)
        return self.__fluorophore_psf[key]

    def __get(self, depth):
        radial = numpy.arange(0.0, self.psf_radial_cutoff, self.__resolution_radial, dtype=float)
        r = radial / self.__resolution_radial
        radial_cutoff = self.psf_radial_cutoff / self.__resolution_radial

        theta = numpy.linspace(0, 90, 91)  #XXX: theta resolution
        z = numpy.linspace(0, radial_cutoff, len(r))
        y = numpy.linspace(0, radial_cutoff, len(r))

        coordinates = _polar2cartesian_coordinates(r, theta, z, y)
        psf_t = numpy.ones_like(theta)

        psf_r = self.get_distribution(radial, depth)
        psf_polar = numpy.array(list(map(lambda x: psf_t * x, psf_r)))
        return _polar2cartesian(psf_polar, coordinates, (len(z), len(y)))

    def __get_normalization(self):
        # Fluorophores Emission Intensity (wave_length)
        I = self.fluoem_norm

        # Photon Transmission Efficiency
        if self.dichroic_switch:
            I = I * 0.01 * self.dichroic_eff

        if self.emission_switch:
            I = I * 0.01 * self.emission_eff

        return numpy.sum(I) * self.psf_normalization

    def get_distribution(self, radial, depth):
        psf = self.get_born_wolf_distribution(radial, depth, self.psf_wavelength)
        return self.__normalization * psf

    @staticmethod
    def get_gaussian_distribution(r, width, normalization):
        #XXX: normalization means I in __get_normalization.

        # For normalization
        # norm = list(map(lambda x: True if x > 1e-4 else False, I))
        norm = (normalization > 1e-4)

        # Ir = sum(list(map(lambda x: x * numpy.exp(-0.5 * (r / self.psf_width[0]) ** 2), norm)))
        # Iz = sum(list(map(lambda x: x * numpy.exp(-0.5 * (z / self.psf_width[1]) ** 2), norm)))
        Ir = norm * numpy.exp(-0.5 * numpy.power(r / width[0], 2))
        Iz = norm * numpy.exp(-0.5 * numpy.power(r / width[1], 2))

        return numpy.array(list(map(lambda x: Ir * x, Iz)))

    @staticmethod
    def get_born_wolf_distribution(r, z, wave_length):
        # set Magnification of optical system
        # M = self.image_magnification

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

        #XXX: z : 1d array_like
        # J0 = numpy.array(list(map(lambda x: j0(x * alpha * rho), r)))
        # print(f'J0.shape = {J0.shape}')
        # Y  = numpy.array(list(map(lambda x: numpy.exp(-2 * 1.j * x * gamma * rho * rho) * rho * drho, z)))
        # print(f'Y.shape = {Y.shape}')
        # I  = numpy.array(list(map(lambda x: x * J0, Y)))
        # print(f'I.shape = {I.shape}')
        # I_sum = I.sum(axis=2)
        # print(f'I_sum.shape = {I_sum.shape}')

        # set PSF
        psf = numpy.array(list(map(lambda x: abs(x) ** 2, I_sum)))
        # print(f'psf.shape = {psf.shape}')
        return psf

    def overlay_signal(self, camera, p_i, pixel_length, normalization=1.0):
        # fluorophore axial position
        if normalization <= 0.0:
            return
        depth = p_i[0]
        signal = self.get(depth)
        self.__overlay_signal(camera, signal, p_i, pixel_length, normalization)

    @staticmethod
    def __overlay_signal(camera, signal, p_i, pixel_length, normalization):
        # m-scale
        # camera pixel
        Nw_pixel, Nh_pixel = camera.shape

        signal_resolution = 1.0e-9  # m  #=> self.psf_radial_cutoff / (len(r) - 1) ???

        # particle position
        _, yi, zi = p_i
        dz = (zi - signal_resolution * signal.shape[0] / 2 + pixel_length * Nw_pixel / 2)
        dy = (yi - signal_resolution * signal.shape[1] / 2 + pixel_length * Nh_pixel / 2)

        imin = math.ceil(dz / pixel_length)
        imax = math.ceil((signal.shape[0] * signal_resolution + dz) / pixel_length) - 1
        jmin = math.ceil(dy / pixel_length)
        jmax = math.ceil((signal.shape[1] * signal_resolution + dy) / pixel_length) - 1
        imin = max(imin, 0)
        imax = min(imax, Nw_pixel)
        jmin = max(jmin, 0)
        jmax = min(jmax, Nh_pixel)

        iarray = numpy.arange(imin, imax + 1)
        ibottom = (iarray[: -1] * pixel_length - dz) / signal_resolution
        ibottom = numpy.maximum(numpy.floor(ibottom).astype(int), 0)
        itop = (iarray[1: ] * pixel_length - dz) / signal_resolution
        itop = numpy.minimum(numpy.floor(itop).astype(int), signal.shape[0])
        irange = numpy.vstack((iarray[: -1], ibottom, itop)).T
        irange = irange[irange[:, 2] > irange[:, 1]]

        jarray = numpy.arange(jmin, jmax)
        jbottom = (jarray[: -1] * pixel_length - dy) / signal_resolution
        jbottom = numpy.maximum(numpy.floor(jbottom).astype(int), 0)
        jtop = (jarray[1: ] * pixel_length - dy) / signal_resolution
        jtop = numpy.minimum(numpy.floor(jtop).astype(int), signal.shape[1])
        jrange = numpy.vstack((jarray[: -1], jbottom, jtop)).T
        jrange = jrange[jrange[:, 2] > jrange[:, 1]]

        for i, i0, i1 in irange:
            for j, j0, j1 in jrange:
                photons = signal[i0: i1, j0: j1].sum()
                if photons > 0:
                    camera[i, j] += photons * normalization

class _EPIFMConfigs:

    def __init__(self):
        pass

    def initialize(self, config, rng=None):
        """Initialize based on the given config.

        Args:
            config (Config): A config object.
            rng (numpy.RandomState, optional): A random number generator
                for initializing this class. Defaults to `None`.

        """
        if rng is None:
            warnings.warn('A random number generator [rng] is not given.')
            rng = numpy.random.RandomState()

        # self.wave_length = numpy.arange(config.psf_min_wave_length, config.psf_max_wave_length, dtype=int)
        self.wave_length = numpy.arange(config.psf_min_wave_length, config.psf_max_wave_length, 1e-9, dtype=float)
        N = len(self.wave_length)
        self.wave_number = 2 * numpy.pi / (self.wave_length / 1e-9)  # 1/nm
        self.excitation_eff = numpy.zeros(N, dtype=float)
        self.dichroic_eff = numpy.zeros(N, dtype=float)
        self.emission_eff = numpy.zeros(N, dtype=float)

        self._set_data('hc_const', config.hc_const)

        self.set_shutter(
            config.shutter_start_time, config.shutter_end_time, config.shutter_time_open,
            config.shutter_time_lapse, config.shutter_switch)

        _log.info('--- Shutter:')
        _log.info('    Start-Time = {} sec'.format(self.shutter_start_time))
        _log.info('    End-Time   = {} sec'.format(self.shutter_end_time))
        _log.info('    Time-open  = {} sec'.format(self.shutter_time_open))
        _log.info('    Time-lapse = {} sec'.format(self.shutter_time_lapse))

        self.set_light_source(
            config.source_type, config.source_wavelength, config.source_flux_density,
            config.source_radius, config.source_angle, config.source_switch)

        _log.info('--- Light Source:{}'.format(self.source_type))
        _log.info('    Wave Length = {} m'.format(self.source_wavelength))
        _log.info('    Beam Flux Density = {} W/cm2'.format(self.source_flux_density))
        _log.info('    1/e2 Radius = {} m'.format(self.source_radius))
        _log.info('    Angle = {} degree'.format(self.source_angle))

        self.set_fluorophore(
            config.fluorophore_type, config.psf_wavelength, config.psf_normalization,
            config.fluorophore_radius, config.psf_width)

        _log.info('--- Fluorophore: {} PSF'.format(self.fluorophore_type))
        _log.info('    Wave Length   =  {} m'.format(self.psf_wavelength))
        _log.info('    Normalization =  {}'.format(self.psf_normalization))
        _log.info('    Fluorophore radius =  {} m'.format(self.fluorophore_radius))
        if hasattr(self, 'psf_width'):
            _log.info('    Lateral Width =  {} m'.format(self.psf_width[0]))
            _log.info('    Axial Width =  {} m'.format(self.psf_width[1]))
        _log.info('    PSF Normalization Factor =  {}'.format(self.psf_normalization))
        _log.info('    Emission  : Wave Length =  {} m'.format(self.psf_wavelength))

        self.set_dichroic_mirror(config.dichroic_mirror, config.dichroic_switch)
        _log.info('--- Dichroic Mirror:')

        self.set_magnification(config.image_magnification)
        _log.info('--- Magnification: x {}'.format(self.image_magnification))

        self.set_detector(
            config.detector_type, config.detector_image_size, config.detector_pixel_length,
            config.detector_exposure_time, config.detector_focal_point, config.detector_qeff,
            config.detector_readout_noise, config.detector_dark_count, config.detector_emgain)

        _log.info('--- Detector:  {}'.format(self.detector_type))
        if hasattr(self, 'detector_image_size'):
            _log.info('    Image Size  =  {} x {}'.format(self.detector_image_size[0], self.detector_image_size[1]))
        _log.info('    Pixel Size  =  {} m/pixel'.format(self.detector_pixel_length))
        _log.info('    Focal Point =  {}'.format(self.detector_focal_point))
        _log.info('    Exposure Time =  {} sec'.format(self.detector_exposure_time))
        if hasattr(self, 'detector_qeff'):
            _log.info('    Quantum Efficiency =  {} %'.format(100 * self.detector_qeff))
        _log.info('    Readout Noise =  {} electron'.format(self.detector_readout_noise))
        _log.info('    Dark Count =  {} electron/sec'.format(self.detector_dark_count))
        _log.info('    EM gain = x {}'.format(self.detector_emgain))
        # _log.info('    Position    =  {}'.format(self.detector_base_position))

        self.set_analog_to_digital_converter(
            config.ADConverter_bit, config.ADConverter_offset, config.ADConverter_fullwell,
            config.ADConverter_fpn_type, config.ADConverter_fpn_count, rng=rng)

        _log.info('--- A/D Converter: %d-bit' % (self.ADConverter_bit))
        # _log.info('    Gain = %.3f electron/count' % (self.ADConverter_gain))
        # _log.info('    Offset =  {} count'.format(self.ADConverter_offset))
        _log.info('    Fullwell =  {} electron'.format(self.ADConverter_fullwell))
        _log.info('    {}-Fixed Pattern Noise: {} count'.format(self.ADConverter_fpn_type, self.ADConverter_fpn_count))

        self.set_illumination_path(config.detector_focal_point, config.detector_focal_norm)
        _log.info('Focal Center: {}'.format(self.detector_focal_point))
        _log.info('Normal Vector: {}'.format(self.detector_focal_norm))

        self.set_excitation_filter(config.excitation_filter, config.excitation_switch)
        _log.info('--- Excitation Filter:')

        self.set_emission_filter(config.emission_filter, config.emission_switch)
        _log.info('--- Emission Filter:')

        # self.set_illumination_path(focal_point=config.detector_focal_point, focal_norm=config.detector_focal_norm)

        self._set_data(
                'fluorophore_psf',
                PointSpreadingFunction(
                    config.psf_radial_cutoff, self.psf_width, config.psf_depth_cutoff, self.fluoem_norm, self.dichroic_switch, self.dichroic_eff, self.emission_switch, self.emission_eff, self.fluorophore_type, self.psf_wavelength, self.psf_normalization))

    def _set_data(self, key, val):
        if val is not None:
            setattr(self, key, val)

    def set_efficiency(self, data):
        data = numpy.array(data, dtype = 'float')
        data = data[data[: , 0] % 1 == 0, :]

        efficiency = numpy.zeros(len(self.wave_length))

        wave_length = numpy.round(self.wave_length / 1e-9).astype(int)  #XXX: numpy.array(dtype=int)
        idx1 = numpy.in1d(wave_length, data[:, 0])
        idx2 = numpy.in1d(numpy.array(data[:, 0]), wave_length)

        efficiency[idx1] = data[idx2, 1]

        return efficiency.tolist()

    def set_shutter(self, start_time=None, end_time=None, time_open=None, time_lapse=None, switch=True):
        self._set_data('shutter_switch', switch)
        self._set_data('shutter_start_time', start_time)
        self._set_data('shutter_end_time', end_time)

        self._set_data('shutter_time_open', time_open or end_time - start_time)
        self._set_data('shutter_time_lapse', time_lapse or end_time - start_time)

    def set_light_source(self, source_type=None, wave_length=None, flux_density=None, radius=None, angle=None, switch=True):
        self._set_data('source_switch', switch)
        self._set_data('source_type', source_type)
        self._set_data('source_wavelength', wave_length)
        self._set_data('source_flux_density', flux_density)
        self._set_data('source_radius', radius)
        self._set_data('source_angle', angle)

    def set_fluorophore(
            self, fluorophore_type=None, wave_length=None, normalization=None, radius=None, width=None):
        self._set_data('fluorophore_type', fluorophore_type)
        self._set_data('fluorophore_radius', radius)
        self._set_data('psf_normalization', normalization)

        if (fluorophore_type == 'Gaussian'):
            N = len(self.wave_length)
            fluorophore_excitation = numpy.zeros(N, dtype=float)
            fluorophore_emission = numpy.zeros(N, dtype=float)
            index = (numpy.abs(self.wave_length / 1e-9 - self.psf_wavelength / 1e-9)).argmin()
            fluorophore_excitation[index] = 100
            fluorophore_emission[index] = 100
            self._set_data('fluoex_eff', fluorophore_excitation)
            self._set_data('fluoem_eff', fluorophore_emission)

            fluorophore_excitation /= sum(fluorophore_excitation)
            fluorophore_emission /= sum(fluorophore_emission)
            self._set_data('fluoex_norm', fluorophore_excitation)
            self._set_data('fluoem_norm', fluorophore_emission)

            self._set_data('psf_wavelength', wave_length)
            self._set_data('psf_width', width)

        else:
            (fluorophore_excitation, fluorophore_emission) = io.read_fluorophore_catalog(fluorophore_type)
            fluorophore_excitation = self.set_efficiency(fluorophore_excitation)
            fluorophore_emission = self.set_efficiency(fluorophore_emission)
            fluorophore_excitation = numpy.array(fluorophore_excitation)
            fluorophore_emission = numpy.array(fluorophore_emission)
            index_ex = fluorophore_excitation.argmax()
            index_em = fluorophore_emission.argmax()
            # index_ex = fluorophore_excitation.index(max(fluorophore_excitation))
            # index_em = fluorophore_emission.index(max(fluorophore_emission))
            fluorophore_excitation[index_ex] = 100
            fluorophore_emission[index_em] = 100
            self._set_data('fluoex_eff', fluorophore_excitation)
            self._set_data('fluoem_eff', fluorophore_emission)

            fluorophore_excitation /= sum(fluorophore_excitation)
            fluorophore_emission /= sum(fluorophore_emission)
            self._set_data('fluoex_norm', fluorophore_excitation)
            self._set_data('fluoem_norm', fluorophore_emission)

            if wave_length is not None:
                warnings.warn('The given wave length [{}] was ignored'.format(wave_length))

            # _log.info('    Excitation: Wave Length =  {} m'.format(self.wave_length[index_ex]))
            self._set_data('psf_wavelength', self.wave_length[index_em])

    def set_magnification(self, magnification=None):
        self._set_data('image_magnification', magnification)

    def set_detector(
            self, detector=None, image_size=None, pixel_length=None, exposure_time=None, focal_point=None,
            QE=None, readout_noise=None, dark_count=None, emgain=None, switch=True):
        self._set_data('detector_switch', switch)
        self._set_data('detector_type', detector)
        self._set_data('detector_image_size', image_size)
        self._set_data('detector_pixel_length', pixel_length)
        self._set_data('detector_exposure_time', exposure_time)
        self._set_data('detector_focal_point', focal_point)
        self._set_data('detector_qeff', QE)
        self._set_data('detector_readout_noise', readout_noise)
        self._set_data('detector_dark_count', dark_count)
        self._set_data('detector_emgain', emgain)
        # self._set_data('detector_base_position', base_position)

    def set_analog_to_digital_converter(self, bit=None, offset=None, fullwell=None, fpn_type=None, fpn_count=None, rng=None):
        self._set_data('ADConverter_bit', bit)
        # self._set_data('ADConverter_offset', offset)
        self._set_data('ADConverter_fullwell', fullwell)
        self._set_data('ADConverter_fpn_type', fpn_type)
        self._set_data('ADConverter_fpn_count', fpn_count)

        # if self.fullwel is not None and self.bit is not None and self.offset is not None:
        #     self._set_data(
        #         'ADConverter_gain',
        #         (self.ADConverter_fullwell - 0.0) / (pow(2.0, self.ADConverter_bit) - self.ADConverter_offset))

        offset, gain = self.calculate_analog_to_digital_converter_gain(offset, rng)
        self._set_data('ADConverter_offset', offset)
        self._set_data('ADConverter_gain', gain)

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

    def set_dichroic_mirror(self, dm=None, switch=True):
        self._set_data('dichroic_switch', switch)
        if dm is not None:
            dichroic_mirror = io.read_dichroic_catalog(dm)
            self._set_data('dichroic_eff', self.set_efficiency(dichroic_mirror))

    def set_excitation_filter(self, excitation=None, switch=True):
        self._set_data('excitation_switch', switch)
        if excitation is not None:
            excitation_filter = io.read_excitation_catalog(excitation)
            self._set_data('excitation_eff', self.set_efficiency(excitation_filter))

    def set_emission_filter(self, emission=None, switch=True):
        self._set_data('emission_switch', switch)
        if emission is not None:
            emission_filter = io.read_emission_catalog(emission)
            self._set_data('emission_eff', self.set_efficiency(emission_filter))

    def set_illumination_path(self, focal_point, focal_norm):
        self._set_data('detector_focal_point', focal_point)
        self._set_data('detector_focal_norm', focal_norm)

    # def set_Illumination_path(self):
    #     # define observational image plane in nm-scale
    #     voxel_size = 2.0 * self.spatiocyte_VoxelRadius

    #     # cell size (nm scale)
    #     Nz = self.spatiocyte_lengths[2] * voxel_size
    #     Ny = self.spatiocyte_lengths[1] * voxel_size
    #     Nx = self.spatiocyte_lengths[0] * voxel_size

    #     # beam center
    #     # b_0 = self.source_center
    #     b_0 = numpy.array(self.detector_focal_point) / 1e-9  # nano meter
    #     x_0, y_0, z_0 = numpy.array([Nx, Ny, Nz]) * b_0

    #     # Incident beam: 1st beam angle to basal region of the cell
    #     theta_in = (self.source_angle / 180.0) * numpy.pi
    #     sin_th1 = numpy.sin(theta_in)
    #     sin2 = sin_th1 * sin_th1

    #     # Index of refraction
    #     n_1 = 1.460 # fused silica
    #     n_2 = 1.384 # cell
    #     n_3 = 1.337 # culture medium

    #     r  = n_2 / n_1
    #     r2 = r * r

    #     if (sin2 / r2 < 1):
    #         raise RuntimeError('Not supported.')

    #         # get cell-shape data
    #         # cell_shape = self.spatiocyte_shape.copy()

    #         # # find cross point of beam center and cell surface
    #         # # rho = numpy.sqrt(Nx * Nx + Ny * Ny)
    #         # rho = Nx

    #         # while (rho > 0):
    #         #     # set beam position
    #         #     #x_b, y_b, z_b = rho*cos_th2, rho*sin_th2 + y_0, z_0
    #         #     x_b, y_b, z_b = rho, y_0, z_0
    #         #     r_b = numpy.array([x_b, y_b, z_b])

    #         #     # evaluate for intersection of beam-line to cell-surface
    #         #     diff  = numpy.sqrt(numpy.sum((cell_shape - r_b) ** 2, axis=1))
    #         #     index = numpy.nonzero(diff < voxel_size)[0]

    #         #     if (len(index) > 0):

    #         #         p_0 = cell_shape[index[0]]
    #         #         x_b, y_b, z_b = p_0

    #         #         f0 = (x_b / Nx, y_b / Ny, z_b / Nz)

    #         #         # evaluate for normal vector of cell-surface
    #         #         diff = numpy.sqrt(numpy.sum((cell_shape - p_0) ** 2, axis=1))
    #         #         k0 = numpy.nonzero(diff < 1.5 * voxel_size)[0]
    #         #         k1 = numpy.nonzero(k0 != diff.argmin())[0]

    #         #         r_n = cell_shape[k0[k1]]

    #         #         # Optimization is definitely required!!
    #         #         f0_norm = numpy.array([0, 0, 0])
    #         #         count = 0
    #         #         for kk in range(len(r_n)):
    #         #             for ii in range(len(r_n)):
    #         #                 for jj in range(len(r_n)):
    #         #                     if (kk!=ii and kk!=jj and ii!=jj):
    #         #                         t = r_n[ii] - r_n[kk]
    #         #                         s = r_n[jj] - r_n[kk]

    #         #                         vec = numpy.cross(s, t)
    #         #                         if (vec[0] < 0): vec = numpy.cross(t, s)
    #         #                         len_vec = numpy.sqrt(numpy.sum(vec**2))
    #         #                         if (len_vec > 0):
    #         #                             f0_norm = f0_norm + vec/len_vec
    #         #                             count += 1

    #         #         f0_norm = f0_norm / count
    #         #         break

    #         #     rho -= voxel_size / 2
    #     else:
    #         f0 = b_0
    #         f0_norm = numpy.array([1, 0, 0])

    #     # set focal position
    #     self.detector_focal_point = f0 * 1e-9  # meter
    #     self.detector_focal_norm  = f0_norm

    #     _log.info('Focal Center: {}'.format(self.detector_focal_point))
    #     _log.info('Normal Vector: {}'.format(self.detector_focal_norm))

    # def set_Detection_path(self):
    #     # # set image scaling factor
    #     # voxel_radius = self.spatiocyte_VoxelRadius

    #     # # set camera's pixel length
    #     # pixel_length = self.detector_pixel_length / self.image_magnification
    #     # # self.image_resolution = pixel_length
    #     # self.image_scaling = pixel_length / (2.0 * voxel_radius)

    #     # # _log.info('Resolution: {} m'.format(self.image_resolution))
    #     # _log.info('Scaling: {}'.format(self.image_scaling))

    #     # Detector PSF
    #     self.set_PSF_detector()

def cylindrical_coordinate(p_i, p_0):
    p_i = numpy.array(p_i)  # Make a copy
    p_0 = numpy.asarray(p_0)

    displacement = p_i - p_0
    radial = numpy.sqrt(displacement[1] ** 2 + displacement[2] ** 2)
    depth = displacement[0]
    return p_i, radial, depth

def _polar2cartesian_coordinates(r, t, x, y):
    X, Y = numpy.meshgrid(x, y)
    new_r = numpy.sqrt(X * X + Y * Y)
    new_t = numpy.arctan2(X, Y)

    ir = interp1d(r, numpy.arange(len(r)), bounds_error=False)
    it = interp1d(t, numpy.arange(len(t)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())
    new_ir[new_r.ravel() > r.max()] = len(r) - 1
    new_ir[new_r.ravel() < r.min()] = 0

    return numpy.array([new_ir, new_it])

def _polar2cartesian(grid, coordinates, shape):
    r = shape[0] - 1
    psf_cart = numpy.empty([2 * r + 1, 2 * r + 1])
    psf_cart[r: , r: ] = map_coordinates(grid, coordinates, order=0).reshape(shape)
    psf_cart[r: , : r] = psf_cart[r: , : r: -1]
    psf_cart[: r, : ] = psf_cart[: r: -1, : ]
    return psf_cart

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
        # if processes is not None and processes > 1:
        #     with Pool(processes) as pool:
        #         signal = pool.map(functools.partial(CMOS.draw_signal, rng=None, emgain=emgain), expected.flatten())
        #     signal = numpy.array(signal).reshape(expected.shape)
        # else:
        #     signal = numpy.zeros(expected.shape, dtype=expected.dtype)
        #     for i in range(expected.shape[0]):
        #         for j in range(expected.shape[1]):
        #             signal[i, j] = CMOS.draw_signal(expected[i, j], emgain, rng)
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
                signal = pool.map(functools.partial(EMCCD.draw_signal, rng=None, emgain=emgain), expected.flatten())
            signal = numpy.array(signal).reshape(expected.shape)
        else:
            signal = numpy.zeros(expected.shape, dtype=expected.dtype)
            for i in range(expected.shape[0]):
                for j in range(expected.shape[1]):
                    signal[i, j] = EMCCD.draw_signal(expected[i, j], emgain, rng)
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
        # if processes is not None and processes > 1:
        #     with Pool(processes) as pool:
        #         signal = pool.map(functools.partial(CCD.draw_signal, rng=None, emgain=emgain), expected.flatten())
        #     signal = numpy.array(signal).reshape(expected.shape)
        # else:
        #     signal = numpy.zeros(expected.shape, dtype=expected.dtype)
        #     for i in range(expected.shape[0]):
        #         for j in range(expected.shape[1]):
        #             signal[i, j] = CCD.draw_signal(expected[i, j], emgain, rng)
        return signal

class EPIFMSimulator:
    '''
    A class of the simulator for Epifluorescence microscopy (EPI).
    '''

    def __init__(self, config=None, rng=None, *, configs=None, effects=None, environ=None):
        """A constructor of EPIFMSimulator.

        Args:
            config (Config, optional): A config object for initializing this class.
            rng (numpy.RandomState, optional): A random number generator
                for initializing this class.

        """
        if configs is not None and effects is not None:
            assert config is None and rng is None
            self.configs = configs
            self.effects = effects
        elif config is not None:
            assert configs is None and effects is None
            self.initialize(config, rng)
        else:
            raise RuntimeError()

        if environ is not None:
            self.environ = environ

    def initialize(self, config=None, rng=None):
        """Initialize EPIFMSimulator.

        Args:
            config (Config, optional): A config object.
            rng (numpy.RandomState, optional): A random number generator.

        """
        if config is None:
            self.__config = Config()
        elif isinstance(config, Config):
            self.__config = config  # copy?
        else:
            self.__config = Config(config)

        self.configs = _EPIFMConfigs()
        self.configs.initialize(self.__config, rng)

        self.effects = PhysicalEffects()
        self.effects.initialize(self.__config)

    def num_frames(self):
        """Return the number of frames within the interval given.

        Returns:
            int: The number of frames available within the interval.

        """
        return math.ceil(
            (self.configs.shutter_end_time - self.configs.shutter_start_time)
            / self.configs.detector_exposure_time)

    def output_frames(
            self, input_data, pathto='./images', data_fmt='image_%07d.npy', true_fmt='true_%07d.npy',
            image_fmt='image_%07d.png', cmin=None, cmax=None, low=None, high=None, cmap=None,
            rng=None, processes=None):
        """Output all images from the given particle data.

        Args:
            input_data (list): An input data. A list of pairs of time and a list of particles.
                Each particle is represented as a list of numbers: a coordinate (a triplet of floats),
                molecule id, species id, and location id (and optionally a state of fluorecent
                and photon budget).
                The number of particles in each frame must be static.
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
            cmap (matplotlib.colors.ColorMap): A color map to visualize images.
                See also `scopyon.image.convert_8bit`.
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

        fluorescence_states = None
        if self.effects.photobleaching_switch:
            fluorescence_states = {}

        results = []
        for frame_index in range(self.num_frames()):
            camera, true_data = self.output_frame(input_data, frame_index, fluorescence_states=fluorescence_states, rng=rng, processes=processes)

            # save photon counts to numpy-binary file
            if data_fmt is not None:
                data_file_name = os.path.join(pathto, data_fmt % (frame_index))
                numpy.save(data_file_name, camera)

            # save true-dataset to numpy-binary file
            if true_fmt is not None:
                true_file_name = os.path.join(pathto, true_fmt % (frame_index))
                numpy.save(true_file_name, true_data)

            # save images to numpy-binary file
            if image_fmt is not None:
                bytedata = convert_8bit(camera[: , : , 1], cmin, cmax, low, high)
                image_file_name = os.path.join(pathto, image_fmt % (frame_index))
                save_image(image_file_name, bytedata, cmap, low, high)

            results.append((camera, true_data))
        return results

    def output_frame(self, input_data, frame_index=0, start_time=None, exposure_time=None, fluorescence_states=None, rng=None, processes=None):
        """Output an image from the given particle data.

        Args:
            input_data (list): An input data. A list of pairs of time and a list of particles.
                Each particle is represented as a list of numbers: a coordinate (x, y, z),
                molecule id, species id, and location id (and optionally a state of fluorecent
                and photon budget).
                The number of particles in each frame must be static.
            frame_index (int, optional): An index of the frame. The value must be 0 and more.
                Defaults to 0.
                See also `EPIFMSimulator.num_frames`.
            start_time (float, optional): A time to open a shutter.
                Defaults to `shutter_start_time` in the configuration.
            exposure_time (float, optional): An exposure time.
                Defaults to `detector_exposure_time` in the configuration.
            rng (numpy.RandomState, optional): A random number generator.

        """
        processes = processes or (1 if self.environ is None else self.environ.processes)

        start_time = start_time or self.configs.shutter_start_time
        # end_time = self.configs.shutter_end_time
        exposure_time = exposure_time or self.configs.detector_exposure_time

        if rng is None:
            _log.info('A random number generator was initialized.')
            rng = numpy.random.RandomState()

        times = numpy.array([t for t, _ in input_data])
        t = start_time + exposure_time * frame_index
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

        # beam position: Assuming beam position = focal point (for temporary)
        p_b = copy.copy(p_0)

        # camera pixels
        Nw_pixel, Nh_pixel = self.configs.detector_image_size  # pixels
        camera_pixel = numpy.zeros((Nw_pixel, Nh_pixel, 2))

        true_data = None

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

            # # define true-dataset in last-frame
            # # [frame-time, m-ID, m-state, p-state, (depth,y0,z0), sqrt(<dr2>)]
            # true_data = numpy.zeros(shape=(len(particles), 7))  #XXX: <== true_data just keeps the last state in the frame
            # true_data[: , 0] = t

            if true_data is None:
                true_data = numpy.zeros(shape=(len(particles), 6))
                true_data[: , 0] = t
            # elif len(particles) > true_data.shape[0]:
            #     particles = numpy.vstack(particles, numpy.zeros(shape=(len(particles) - true_data.shape[0], 7)))

            # loop for particles
            for particle_j, true_data_j in zip(particles, true_data):
                self.__overlay_molecule_plane(camera_pixel[: , : , 0], particle_j, p_b, p_0, true_data_j, unit_time, rng, fluorescence_states)

        true_data[: , 2: 6] /= exposure_time

        # apply detector effects
        camera, true_data = self.__detector_output(rng, camera_pixel, true_data, processes=processes)
        return (camera, true_data)

    def __overlay_molecule_plane(self, camera, particle_i, _p_b, p_0, true_data_i, unit_time, rng=None, fluorescence_states=None):
        # m-scale
        # p_b (ndarray): beam position (assumed to be the same with focal center), but not used.
        # p_0 (ndarray): focal center

        # particles coordinate, species and lattice-IDs
        # x, y, z, m_id, s_id, _, p_state, _ = particle_i
        x, y, z, m_id, p_state = particle_i

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
            if self.effects.photobleaching_switch:
                if m_id not in fluorescence_states:
                    amplitude0, _ = self.snells_law()
                    N_emit0 = self.__get_emit_photons(amplitude0, 1.0)
                    fluorescence_states[m_id] = self.get_photon_budget(
                            N_emit0, self.effects.photobleaching_half_life, rng=rng)  #XXX: rng is only required here
                budget = fluorescence_states[m_id] - N_emit
                if budget <= 0:
                    budget = 0
                    p_state = 0.0  #XXX
                fluorescence_states[m_id] = budget

        normalization = p_state * N_emit / (4.0 * numpy.pi)

        # add signal matrix to image plane
        pixel_length = self.configs.detector_pixel_length / self.configs.image_magnification
        self.configs.fluorophore_psf.overlay_signal(camera, p_i - p_0, pixel_length, normalization)

        # set true-dataset
        true_data_i[1] = m_id # molecule-ID
        true_data_i[2] += unit_time * p_state # photon-state
        true_data_i[3] += unit_time * p_i[2] # Y-coordinate in the image-plane
        true_data_i[4] += unit_time * p_i[1] # X-coordinate in the image-plane
        true_data_i[5] += unit_time * depth # Depth from focal-plane

    def __get_emit_photons(self, amplitude, unit_time):
        # Absorption coeff [1/(cm M)]
        abs_coeff = self.effects.abs_coefficient
        # Quantum yield
        QY = self.effects.quantum_yield
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

        # (plank const) * (speed of light) [joules meter]
        hc = self.configs.hc_const

        # Illumination: Assume that uniform illumination (No gaussian)
        # flux density [W/cm2 (joules/sec/m2)]
        P_0 = self.configs.source_flux_density * 1e+4

        # single photon energy
        wave_length = self.configs.source_wavelength  # m
        E_wl = hc / wave_length

        # photon flux density [photons/sec/m2]
        N_0 = P_0 / E_wl

        # Incident beam: Amplitude
        A2_Is = N_0
        A2_Ip = N_0

        # incident beam angle
        angle = self.configs.source_angle
        theta_in = (angle / 180.) * numpy.pi

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

    def __detector_output(self, rng, camera_pixel, true_data, processes=1):
        Nw_pixel, Nh_pixel = self.configs.detector_image_size  # pixels
        pixel_length = self.configs.detector_pixel_length / self.configs.image_magnification  # m-scale
        _, y0, z0 = self.configs.detector_focal_point  # m-scale
        _log.info('scaling [m/pixel]: {}'.format(pixel_length))
        _log.info('center (width, height): {} {}'.format(z0, y0))
        true_data[: , 3] = (true_data[: , 3] - (z0 - Nw_pixel * pixel_length / 2)) / pixel_length
        true_data[: , 4] = (true_data[: , 4] - (y0 - Nh_pixel * pixel_length / 2)) / pixel_length

        ## Detector: Quantum Efficiency
        # index = int(self.configs.psf_wavelength / 1e-9) - int(self.configs.wave_length[0] / 1e-9)
        # QE = self.configs.detector_qeff[index]
        QE = self.configs.detector_qeff
        ## get signal (photons)
        photons = camera_pixel[:, :, 0]
        ## get constant background (photoelectrons)
        photons += self.effects.background_mean
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
        return camera_pixel, true_data

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
