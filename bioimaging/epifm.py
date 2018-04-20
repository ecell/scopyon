import sys
import shutil
import copy
import math
import operator
import csv
import string
import ctypes
import multiprocessing
import warnings
import os

import numpy

from scipy.special import j0, i1, i1e
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from .effects import PhysicalEffects
from . import io
from .config import Config

 # import numpy.random as rng

from logging import getLogger
_log = getLogger(__name__)


class EPIFMConfigs:
    '''
    EPIFM Configuration

        Wide-field Gaussian Profile
            +
        Detector: EMCCD/CMOS
    '''

    def __init__(self):
        pass

    def initialize(self, config, rng=None):
        if rng is None:
            warnings.warn('A random number generator [rng] is not given.')
            rng = numpy.random.RandomState()

        self.ADConverter_fpn_type = None
        # self.emission_switch = False
        self.hc_const = 2.00e-25

        self.radial = numpy.arange(1000, dtype=float)
        self.depth = numpy.arange(1000, dtype=float)
        self.wave_length = numpy.arange(300, 1000, dtype=int)
        N = len(self.wave_length)
        self.wave_number = 2 * numpy.pi / self.wave_length
        self.excitation_eff = numpy.zeros(N, dtype=float)
        self.dichroic_eff = numpy.zeros(N, dtype=float)
        self.emission_eff = numpy.zeros(N, dtype=float)

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
        _log.info('    Wave Length = {} nm'.format(self.source_wavelength))
        _log.info('    Beam Flux Density = {} W/cm2'.format(self.source_flux_density))
        _log.info('    1/e2 Radius = {} m'.format(self.source_radius))
        _log.info('    Angle = {} degree'.format(self.source_angle))

        self.set_fluorophore(
            config.fluorophore_type, config.psf_wavelength, config.psf_normalization,
            config.fluorophore_radius, config.psf_width, config.psf_cutoff, config.psf_file_name_format)

        _log.info('--- Fluorophore: {} PSF'.format(self.fluorophore_type))
        _log.info('    Wave Length   =  {} nm'.format(self.psf_wavelength))
        _log.info('    Normalization =  {}'.format(self.psf_normalization))
        _log.info('    Fluorophore radius =  {} nm'.format(self.fluorophore_radius))
        if hasattr(self, 'psf_width'):
            _log.info('    Lateral Width =  {} nm'.format(self.psf_width[0]))
            _log.info('    Axial Width =  {} nm'.format(self.psf_width[1]))
        _log.info('    PSF Normalization Factor =  {}'.format(self.psf_normalization))
        _log.info('    Emission  : Wave Length =  {} nm'.format(self.psf_wavelength))

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

        self._set_data('fluorophore_psf', self.get_PSF_detector())

    def _set_data(self, key, val):
        if val is not None:
            setattr(self, key, val)

    def set_efficiency(self, data):
        data = numpy.array(data, dtype = 'float')
        data = data[data[: , 0] % 1 == 0, :]

        efficiency = numpy.zeros(len(self.wave_length))
        idx1 = numpy.in1d(numpy.array(self.wave_length), data[:, 0])
        idx2 = numpy.in1d(numpy.array(data[:, 0]), self.wave_length)

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
            self, fluorophore_type=None, wave_length=None, normalization=None, radius=None, width=None, cutoff=None, file_name_format=None):
        self._set_data('fluorophore_type', fluorophore_type)
        self._set_data('fluorophore_radius', radius)
        self._set_data('psf_normalization', normalization)
        self._set_data('psf_file_name_format', file_name_format)

        if (fluorophore_type == 'Gaussian'):
            N = len(self.wave_length)
            fluorophore_excitation = numpy.zeros(N, dtype=float)
            fluorophore_emission = numpy.zeros(N, dtype=float)
            index = (numpy.abs(self.wave_length - self.psf_wavelength)).argmin()
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
            self._set_data('psf_cutoff', cutoff)

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

            # _log.info('    Excitation: Wave Length =  {} nm'.format(self.wave_length[index_ex]))
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

        if FPN_type is None:
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
            raise ValueError("FPN type [{}] is invalid ['pixel', 'column' or None]".format(FPN_type))

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
    #     b_0 = self.detector_focal_point
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
    #     self.detector_focal_point = f0
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

    def get_PSF_detector(self):
        r = self.radial
        z = self.depth

        wave_length = self.psf_wavelength

        # Fluorophores Emission Intensity (wave_length)
        I = self.fluoem_norm

        # Photon Transmission Efficiency
        if self.dichroic_switch:
            I = I * 0.01 * self.dichroic_eff

        if self.emission_switch:
            I = I * 0.01 * self.emission_eff

        # For normalization
        # norm = list(map(lambda x: True if x > 1e-4 else False, I))
        norm = (I > 1e-4)

        # PSF: Fluorophore

        if self.fluorophore_type == 'Gaussian':
            N0 = self.psf_normalization
            # Ir = sum(list(map(lambda x: x * numpy.exp(-0.5 * (r / self.psf_width[0]) ** 2), norm)))
            # Iz = sum(list(map(lambda x: x * numpy.exp(-0.5 * (z / self.psf_width[1]) ** 2), norm)))
            Ir = norm * numpy.exp(-0.5 * numpy.power(r / self.psf_width[0], 2))
            Iz = norm * numpy.exp(-0.5 * numpy.power(r / self.psf_width[1], 2))

            psf_fl = numpy.sum(I) * N0 * numpy.array(list(map(lambda x: Ir * x, Iz)))

        else:
            # make the norm and wave_length array shorter
            psf_fl = numpy.sum(I) * self.get_PSF_fluorophore(r, z, wave_length)

        # self.fluorophore_psf = psf_fl
        # self._set_data('fluorophore_psf', psf_fl)
        return psf_fl

    def get_PSF_fluorophore(self, r, z, wave_length):
        # set Magnification of optical system
        M = self.image_magnification

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

        J0 = numpy.array(list(map(lambda x: j0(x * alpha * rho), r)))
        Y  = numpy.array(list(map(lambda x: numpy.exp(-2 * 1.j * x * gamma * rho * rho) * rho * drho, z)))
        I  = numpy.array(list(map(lambda x: x * J0, Y)))
        I_sum = I.sum(axis=2)

        # set normalization factor
        Norm = self.psf_normalization

        # set PSF
        psf = Norm * numpy.array(list(map(lambda x: abs(x) ** 2, I_sum)))
        return psf

def rotate_coordinate(p_i, p_0):
    x_0, y_0, z_0 = p_0
    x_i, y_i, z_i = p_i

    # Rotation of focal plane
    cos_th0 = 1
    sin_th0 = numpy.sqrt(1 - cos_th0 * cos_th0)

    # Rotational matrix along z-axis
    #Rot = numpy.matrix([[cos_th, -sin_th, 0], [sin_th, cos_th, 0], [0, 0, 1]])
    Rot = numpy.matrix([[cos_th0, -sin_th0, 0], [sin_th0, cos_th0, 0], [0, 0, 1]])

    # Vector of focal point to particle position
    vec = p_i - p_0
    len_vec = numpy.sqrt(numpy.sum(vec * vec))

    # Rotated particle position
    v_rot = Rot * vec.reshape((3, 1))
    # p_i = numpy.array(v_rot).ravel() + p_0
    newp_i = numpy.array(v_rot).ravel() + p_0

    # Normal vector of the focal plane
    q_0 = numpy.array([0.0, y_0, 0.0])
    q_1 = numpy.array([0.0, 0.0, z_0])
    R_q0 = numpy.sqrt(numpy.sum(q_0 * q_0))
    R_q1 = numpy.sqrt(numpy.sum(q_1 * q_1))

    q0_rot = Rot * q_0.reshape((3, 1))
    q1_rot = Rot * q_1.reshape((3, 1))

    norm_v = numpy.cross(q0_rot.ravel(), q1_rot.ravel()) / (R_q0 * R_q1)

    # Radial distance and depth to focal plane
    cos_0i = numpy.sum(norm_v * vec) / (1.0 * len_vec)
    sin_0i = numpy.sqrt(1 - cos_0i * cos_0i)

    focal_depth  = abs(len_vec * cos_0i)
    focal_radial = abs(len_vec * sin_0i)

    return newp_i, focal_radial, focal_depth

def polar2cartesian_coordinates(r, t, x, y):
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

def polar2cartesian(grid, coordinates, shape):
    r = shape[0] - 1
    psf_cart = numpy.empty([2 * r + 1, 2 * r + 1])
    psf_cart[r: , r: ] = map_coordinates(grid, coordinates, order=0).reshape(shape)
    psf_cart[r: , : r] = psf_cart[r: , : r: -1]
    psf_cart[: r, : ] = psf_cart[: r: -1, : ]
    return psf_cart

class EPIFMVisualizer:
    '''
    EPIFM Visualization class of e-cell simulator
    '''

    def __init__(self, nprocs=1):
        self.__nprocs = nprocs

    def initialize(self, config=None, rng=None):
        if config is None:
            self.__config = Config()
        elif isinstance(config, Config):
            self.__config = config  # copy?
        else:
            self.__config = Config(config)

        self.configs = EPIFMConfigs()
        self.configs.initialize(self.__config, rng)

        self.effects = PhysicalEffects()
        self.effects.initialize(self.__config)

    def num_frames(self):
        return math.ceil(
            (self.configs.shutter_end_time - self.configs.shutter_start_time)
            / self.configs.detector_exposure_time)

    def get_cell_size(self, lengths):
        return numpy.array([int(x) for x in lengths])

    def get_focal_center(self, cell_size):
        # focal point
        return cell_size * self.configs.detector_focal_point

    def apply_photophysics_effects(self, dataset, rng=None):
        end_time = self.configs.shutter_end_time

        N_particles = [len(particles) for _, particles in dataset.data]
        if min(N_particles) != max(N_particles):
            raise RuntimeError('The number of particles must be static during the simulation')
        N_particles = N_particles[0]

        # get focal point
        cell_size = self.get_cell_size(dataset.lengths)
        p_0 = self.get_focal_center(cell_size)

        # beam position: Assuming beam position = focal point (for temporary)
        p_b = copy.copy(p_0)

        # Snell's law
        amplitude0, penet_depth = self.snells_law(p_0, p_0)

        # # copy shape-file
        # csv_shape = self.configs.spatiocyte_file_directory + '/pt-shape.csv'
        # shutil.copyfile(csv_shape, output_file_dir + '/pt-shape.csv')

        # determine photon budgets
        time_array = numpy.array([t for t, _ in dataset.data])
        time_array -= self.configs.shutter_start_time

        N_emit0 = self.get_emit_photons(amplitude0, 1.0)
        fluorescence_state, fluorescence_budget = self.effects.get_photophysics_for_epifm(
            time_array, N_emit0, N_particles, rng)

        new_data = []
        molecule_states = numpy.zeros(shape=(N_particles))
        for k in range(len(dataset.data)):
            # read input file
            # csv_file_path = self.configs.spatiocyte_file_directory + '/pt-%09d.0.csv' % (count_array[k])
            # csv_list = list(csv.reader(open(csv_file_path, 'r')))
            # dataset = numpy.array(csv_list)
            t, particles = dataset.data[k]
            next_time = dataset.data[k + 1][0] if k + 1 < len(dataset.data) else end_time
            unit_time = next_time - t

            # set molecular-states (unbound-bound)
            # self.set_molecular_states(k, dataset)
            self.initialize_molecular_states(molecule_states, k, particles)

            N_emit0 = self.get_emit_photons(amplitude0, unit_time)

            # set photobleaching-dataset arrays
            self.update_fluorescence_photobleaching(fluorescence_state, fluorescence_budget, k, particles, p_0, unit_time)

            # get new-dataset
            new_state = self.get_new_state(molecule_states, fluorescence_state, fluorescence_budget, k, N_emit0)
            # new_dataset = numpy.column_stack((particles, state_stack))

            new_particles = []
            for particle_j, (new_p_state, new_cyc_id) in zip(particles, new_state):
                (coordinate, m_id, s_id, l_id, p_state, cyc_id) = particle_j
                new_particles.append((coordinate, m_id, s_id, l_id, new_p_state, new_cyc_id))
            new_data.append((t, new_particles))

            # # write output file
            # output_file = output_file_dir + '/pt-%09d.0.csv' % (count_array[k])

            # with open(output_file, 'w') as f:
            #     writer = csv.writer(f)
            #     writer.writerows(new_dataset)

        keyval = dataset._asdict()
        keyval['data'] = new_data
        return dataset.__class__(**keyval)

    def initialize_molecular_states(self, states, count, data):
        # reset molecule-states
        states.fill(0.0)

        # loop for particles
        for j, particle_j in enumerate(data):
            (coordinate, m_id, s_id, l_id, p_state, cyc_id) = particle_j

            # set particle position
            p_i = numpy.array(coordinate) / 1e-9

            # set molecule-states
            states[m_id] = int(s_id)

    def update_fluorescence_photobleaching(self, fluorescence_state, fluorescence_budget, count, data, focal_center, unit_time):
        if len(data) == 0:
            return

        if self.get_nprocs() != 1:
            raise RuntimeError('Not supported.')
            #XXX: # set arrays for photobleaching-state and photon-budget
            #XXX: state_pb = {}
            #XXX: budget = {}

            #XXX: num_processes = min(multiprocessing.cpu_count(), len(data))
            #XXX: n, m = divmod(len(data), num_processes)

            #XXX: chunks = [n + 1 if i < m else n for i in range(num_processes)]

            #XXX: processes = []
            #XXX: start_index = 0

            #XXX: for chunk in chunks:
            #XXX:     stop_index = start_index + chunk
            #XXX:     p, c = multiprocessing.Pipe()
            #XXX:     process = multiprocessing.Process(target=self.get_photobleaching_dataset_process,
            #XXX:                             args=(count, data[start_index:stop_index], focal_center, c))
            #XXX:     processes.append((p, process))
            #XXX:     start_index = stop_index

            #XXX: for _, process in processes:
            #XXX:     process.start()

            #XXX: for pipe, process in processes:
            #XXX:     new_state_pb, new_budget = pipe.recv()
            #XXX:     state_pb.update(new_state_pb)
            #XXX:     budget.update(new_budget)
            #XXX:     process.join()

        else:
            state_pb, budget = self.get_fluorescence_photobleaching(fluorescence_state, fluorescence_budget, count, data, focal_center, unit_time)

        # reset global-arrays for photobleaching-state and photon-budget
        for key, value in state_pb.items():
            fluorescence_state[key, count] = state_pb[key]
            fluorescence_budget[key] = budget[key]
            # self.effects.fluorescence_state[key,count] = state_pb[key]
            # self.effects.fluorescence_budget[key] = budget[key]

    def get_fluorescence_photobleaching(self, fluorescence_state, fluorescence_budget, count, data, focal_center, unit_time):
        # get focal point
        p_0 = focal_center

        # set arrays for photobleaching-states and photon-budget
        result_state_pb = {}
        result_budget = {}

        # loop for particles
        for j, particle_j in enumerate(data):
            (coordinate, m_id, s_id, l_id, p_state, cyc_id) = particle_j

            # set particle position
            p_i = numpy.array(coordinate) / 1e-9

            # Snell's law
            amplitude, penet_depth = self.snells_law(p_i, p_0)

            # particle coordinate in real(nm) scale
            p_i, radial, depth = rotate_coordinate(p_i, p_0)

            state_j = 1  # particles given is always observable. already filtered when read

            # get exponential amplitude (only for observation at basal surface)
            # amplitide = amplitude * numpy.exp(-depth / pent_depth)

            # get the number of emitted photons
            N_emit = self.get_emit_photons(amplitude, unit_time)

            # get global-arrays for photobleaching-state and photon-budget
            state_pb = fluorescence_state[m_id, count]
            budget = fluorescence_budget[m_id]

            # reset photon-budget
            photons = budget - N_emit * state_j

            if (photons > 0):
                budget = photons
                state_pb = state_j
            else:
                budget = 0
                state_pb = 0

            result_state_pb[m_id] = state_pb
            result_budget[m_id] = budget

        return result_state_pb, result_budget

    #XXX: def get_photobleaching_dataset_process(self, count, dataset, voxel_radius, lengths, pipe):
    #XXX:     pipe.send(self.get_photobleaching_dataset(count, dataset, voxel_radius, lengths))

    def get_new_state(self, molecule_states, fluorescence_state, fluorescence_budget, count, N_emit0):
        state_mo = molecule_states
        state_pb = fluorescence_state[: , count]
        budget = fluorescence_budget

        # set additional arrays for new-dataset
        new_state_pb = state_pb[state_mo > 0]
        new_budget = budget[state_mo > 0]

        # set new-dataset
        state_stack = numpy.column_stack((new_state_pb, (new_budget / N_emit0).astype('int')))
        return state_stack

    def overlay_molecule_plane(self, cell, j, particle_j, p_b, p_0, true_data, unit_time, fluo_psfs=None):
        # particles coordinate, species and lattice-IDs
        coordinate, m_id, s_id, _, p_state, _ = particle_j

        p_i = numpy.array(coordinate) / 1e-9

        # Snell's law
        amplitude, penet_depth = self.snells_law(p_i, p_0)

        # particles coordinate in real(nm) scale
        p_i, radial, depth = rotate_coordinate(p_i, p_0)

        # get exponential amplitude (only for TIRFM-configuration)
        amplitude = amplitude * numpy.exp(-depth / penet_depth)

        # get signal matrix
        signal = self.get_signal(amplitude, radial, depth, p_state, unit_time, fluo_psfs)

        # add signal matrix to image plane
        self.overlay_signal(cell, signal, p_i, p_0)
        # self.overwrite_signal(cell, signal, p_i)

        # set true-dataset
        true_data[j, 1] = m_id # molecule-ID
        true_data[j, 2] = int(s_id) # sid_index # molecular-state
        true_data[j, 3] = p_state # photon-state
        true_data[j, 4] = p_i[2] # Y-coordinate in the image-plane
        true_data[j, 5] = p_i[1] # X-coordinate in the image-plane
        true_data[j, 6] = depth  # Depth from focal-plane

    def overlay_signal(self, cell, signal, p_i, p_0):
        # particle position
        _, yi, zi = p_i  # rotated particle position
        _, y0, z0 = p_0  # focal center

        z0_from = math.floor((zi - signal.shape[0] / 2) - (z0 - cell.shape[0] / 2))
        zi_from = 0
        w = signal.shape[0]
        if z0_from > cell.shape[0] or z0_from + w < 0:
            return
        elif z0_from < 0:
            w += z0_from
            zi_from = -z0_from
            z0_from = 0
        elif z0_from + w > cell.shape[0]:
            w = cell.shape[0] - z0_from

        y0_from = math.floor((yi - signal.shape[1] / 2) - (y0 - cell.shape[1] / 2))
        yi_from = 0
        h = signal.shape[1]
        if y0_from > cell.shape[1] or y0_from + h < 0:
            return
        elif y0_from < 0:
            h += y0_from
            yi_from = -y0_from
            y0_from = 0
        elif y0_from + h > cell.shape[1]:
            h = cell.shape[1] - y0_from

        assert 0 <= z0_from and z0_from + w <= cell.shape[0]
        assert 0 <= y0_from and y0_from + h <= cell.shape[1]
        assert 0 <= zi_from and zi_from + w <= signal.shape[0]
        assert 0 <= yi_from and yi_from + h <= signal.shape[1]
        print((z0_from, y0_from, zi_from, yi_from, w, h, cell.shape, signal.shape))

        cell[z0_from: z0_from + w, y0_from: y0_from + h] += signal[zi_from: zi_from + w, yi_from: yi_from + h]

    def overwrite_signal(self, cell, signal, p_i):
        # particle position
        x_i, y_i, z_i = p_i

        flag = True

        # z-axis
        Nz_cell  = len(cell)
        Nz_signal = len(signal)
        Nr = len(self.configs.radial)

        z_to   = z_i + Nr
        z_from = z_i - Nr

        if (z_to > Nz_cell):
            dz_to = z_to - Nz_cell
            z0_to = int(Nz_cell)
            zi_to = int(Nz_signal - dz_to)
        elif (z_to > 0 and z_to < Nz_cell):
            dz_to = Nz_cell - z_to
            z0_to = int(Nz_cell - dz_to)
            zi_to = int(Nz_signal)
        else:
            flag = False

        if (z_from < 0):
            dz_from = abs(z_from)
            z0_from = 0
            zi_from = int(dz_from)
        elif (z_from > 0 and z_from < Nz_cell):
            dz_from = z_from
            z0_from = int(dz_from)
            zi_from = 0
        else:
            flag = False

        # y-axis
        Ny_cell  = cell.size/Nz_cell
        Ny_signal = signal.size/Nz_signal

        y_to   = y_i + Nr
        y_from = y_i - Nr

        if (y_to > Ny_cell):
            dy_to = y_to - Ny_cell
            y0_to = int(Ny_cell)
            yi_to = int(Ny_signal - dy_to)
        elif (y_to > 0 and y_to < Ny_cell):
            dy_to = Ny_cell - y_to
            y0_to = int(Ny_cell - dy_to)
            yi_to = int(Ny_signal)
        else:
            flag = False

        if (y_from < 0):
            dy_from = abs(y_from)
            y0_from = 0
            yi_from = int(dy_from)
        elif (y_from > 0 and y_from < Ny_cell):
            dy_from = y_from
            y0_from = int(dy_from)
            yi_from = 0
        else:
            flag = False

        # signal plane in the cellular plane
        if (flag == True):

            # adjust index
            ddy = (y0_to - y0_from) - (yi_to - yi_from)
            ddz = (z0_to - z0_from) - (zi_to - zi_from)

            if (ddy != 0): y0_to = y0_to - ddy
            if (ddz != 0): z0_to = z0_to - ddz

            # add to cellular plane
            cell[z0_from:z0_to, y0_from:y0_to] += signal[zi_from:zi_to, yi_from:yi_to]

    def get_molecule_plane(self, cell, j, particle_j, p_b, p_0, true_data, unit_time, fluo_psfs=None):
        # particles coordinate, species and lattice-IDs
        coordinate, m_id, s_id, _, p_state, _ = particle_j

        p_i = numpy.array(coordinate) / 1e-9

        # Snell's law
        amplitude, penet_depth = self.snells_law(p_i, p_0)

        # particles coordinate in real(nm) scale
        p_i, radial, depth = rotate_coordinate(p_i, p_0)

        # get exponential amplitude (only for TIRFM-configuration)
        amplitude = amplitude * numpy.exp(-depth / penet_depth)

        # get signal matrix
        signal = self.get_signal(amplitude, radial, depth, p_state, unit_time, fluo_psfs)

        # add signal matrix to image plane
        self.overwrite_signal(cell, signal, p_i)

        # set true-dataset
        true_data[j, 1] = m_id # molecule-ID
        true_data[j, 2] = int(s_id) # sid_index # molecular-state
        true_data[j, 3] = p_state # photon-state
        true_data[j, 4] = p_i[2] # Y-coordinate in the image-plane
        true_data[j, 5] = p_i[1] # X-coordinate in the image-plane
        true_data[j, 6] = depth  # Depth from focal-plane

    def get_signal(self, amplitude, radial, depth, p_state, unit_time, fluo_psfs=None):
        # fluorophore axial position
        fluo_depth = depth if depth < len(self.configs.depth) else -1

        # get fluorophore PSF
        psf_depth = (fluo_psfs or self.fluo_psf)[int(fluo_depth)]

        # get the number of photons emitted
        N_emit = self.get_emit_photons(amplitude, unit_time)

        # get signal
        signal = p_state * N_emit / (4.0 * numpy.pi) * psf_depth

        return signal

    def get_emit_photons(self, amplitude, unit_time):
        # Absorption coeff [1/(cm M)]
        abs_coeff = self.effects.abs_coefficient

        # Quantum yield
        QY = self.effects.quantum_yield

        # Abogadoro's number
        Na = self.effects.avogadoros_number

        # Cross-section [m2]
        x_sec = numpy.log(10) * abs_coeff * 0.1 / Na

        # get the number of absorption photons: [#/(m2 sec)]*[m2]*[sec]
        n_abs = amplitude * x_sec * unit_time

        # Beer-Lamberts law: A = log(I0 / I) = abs coef. * concentration * path length ([m2]*[#/m3]*[m])
        fluorophore_radius = self.configs.fluorophore_radius * 1e-9
        fluorophore_volume = (4.0 / 3.0) * numpy.pi * numpy.power(fluorophore_radius, 3)
        fluorophore_depth  = 2.0 * fluorophore_radius

        A = (abs_coeff * 0.1 / Na) * (1.0 / fluorophore_volume) * fluorophore_depth

        # get the number of photons emitted
        n_emit = QY * n_abs * (1.0 - numpy.power(10.0, -A))

        return n_emit

    def snells_law(self, p_i, p_0):
        x_0, y_0, z_0 = p_0
        x_i, y_i, z_i = p_i

        # (plank const) * (speed of light) [joules meter]
        hc = self.configs.hc_const

        # Illumination: Assume that uniform illumination (No gaussian)
        # flux density [W/cm2 (joules/sec/m2)]
        P_0 = self.configs.source_flux_density*1e+4

        # single photon energy
        wave_length = self.configs.source_wavelength*1e-9
        E_wl = hc/wave_length

        # photon flux density [photons/sec/m2]
        N_0 = P_0/E_wl

        # Incident beam: Amplitude
        A2_Is = N_0
        A2_Ip = N_0

        # incident beam angle
        angle = self.configs.source_angle
        theta_in = (angle/180.)*numpy.pi

        sin_th1 = numpy.sin(theta_in)
        cos_th1 = numpy.cos(theta_in)

        sin = sin_th1
        cos = cos_th1
        sin2 = sin**2
        cos2 = cos**2

        # index of refraction
        n_1 = 1.46  # fused silica
        n_2 = 1.384 # cell
        n_3 = 1.337 # culture medium

        r  = n_2/n_1
        r2 = r**2

        # Epi-illumination at apical surface
        if (sin2/r2 < 1):
            raise RuntimeError('Not supported.')

            # # Refracted beam: 2nd beam angle to basal region of the cell
            # sin_th2 = (n_1/n_2)*sin_th1
            # cos_th2 = numpy.sqrt(1 - sin_th2**2)
            # beam = numpy.array([cos_th2, sin_th2, 0])

            # # Normal vector: Perpendicular to apical surface of the cell
            # voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius
            # cell_shape = self.configs.spatiocyte_shape.copy()

            # diff = numpy.sqrt(numpy.sum((cell_shape - p_i*1e-9)**2, axis=1))
            # k0 = numpy.nonzero(diff < 1.5*voxel_size)[0]
            # k1 = numpy.nonzero(k0 != diff.argmin())[0]

            # r_n = cell_shape[k0[k1]]

            # # Optimization is definitely required!!
            # norm = numpy.array([0, 0, 0])
            # count = 0
            # for kk in range(len(r_n)):
            #     for ii in range(len(r_n)):
            #         for jj in range(len(r_n)):
            #             if (kk!=ii and kk!=jj and ii!=jj):
            #                 t = r_n[ii] - r_n[kk]
            #                 s = r_n[jj] - r_n[kk]

            #                 vec = numpy.cross(s, t)
            #                 if (vec[0] < 0): vec = numpy.cross(t, s)
            #                 len_vec = numpy.sqrt(numpy.sum(vec**2))
            #                 if (len_vec > 0):
            #                     norm = norm + vec/len_vec
            #                     count += 1

            # norm = norm/count

            # # Plane to separate between apical and basal cell-surface regions
            # v_0 = numpy.array([voxel_size/1e-9, y_0, 0])
            # v_1 = numpy.array([voxel_size/1e-9, 0, z_0])
            # v_2 = numpy.cross(v_0, v_1)
            # len_v2 = numpy.sqrt(numpy.sum(v_2**2))
            # plane_vec = v_2/len_v2

            # # Apical region (>0) and Basal region (<0)
            # a, b, c = plane_vec
            # plane_eqn = a*(x_i - voxel_size/1e-9) + b*y_i + c*z_i

            # # check the direction of normal vector at each regions
            # check = numpy.dot(plane_vec, norm)

            # if (plane_eqn < 0 and check > 0):
            #     norm = -norm
            # elif (plane_eqn > 0 and check < 0):
            #     norm = -norm

            # # Incident beam: 3rd beam angle to apical surface of the cell
            # #cos_th3 = numpy.dot(beam, norm)
            # norm_x, norm_y, norm_z = norm
            # len_norm_xy = numpy.sqrt(norm_x**2 + norm_y**2)
            # norm_xy = numpy.array([norm_x, norm_y, 0])/len_norm_xy
            # cos_th3 = numpy.dot(beam, norm_xy)

            # if (cos_th3 > 0 and plane_eqn > 0):

            #     # Incident beam to apical surface: amplitude
            #     cosT = numpy.sqrt(1 - sin2/r2)

            #     A2_Ip = A2_Ip*(2*cos/(cosT + r*cos))**2
            #     A2_Is = A2_Is*(2*cos/(r*cosT + cos))**2

            #     # Incident beam to apical surface: 3rd beam angle
            #     cos = cos_th3
            #     sin = numpy.sqrt(1 - cos**2)
            #     sin2 = sin**2
            #     cos2 = cos**2

            #     r  = n_3/n_2 # must be < 1
            #     r2 = r**2

            #     if (sin2/r2 > 1):
            #         # Evanescent field: Amplitude and Penetration depth
            #         # Assume that the s-polar direction is parallel to y-axis
            #         A2_x = A2_Ip*(4*cos2*(sin2 - r2)/(r2**2*cos2 + sin2 - r2))
            #         A2_y = A2_Is*(4*cos2/(1 - r2))
            #         A2_z = A2_Ip*(4*cos2*sin2/(r2**2*cos2 + sin2 - r2))

            #         A2_Tp = A2_x + A2_z
            #         A2_Ts = A2_y

            #         #penetration_depth = wave_length/(4.0*numpy.pi*numpy.sqrt(n_2**2*sin2 - n_3**2))
            #     else:
            #         # Epi-fluorescence field: Amplitude and Penetration depth
            #         cosT = numpy.sqrt(1 - sin2/r2)

            #         A2_Tp = A2_Ip*(2*cos/(cosT + r*cos))**2
            #         A2_Ts = A2_Is*(2*cos/(r*cosT + cos))**2

            #         #penetration_depth = float('inf')
            # else:
            #     # Epi-fluorescence field: Amplitude and Penetration depth
            #     A2_Tp = A2_Ip
            #     A2_Ts = A2_Is

            #     #penetration_depth = float('inf')

            # # for temp
            # penetration_depth = float('inf')
        else:
            # TIRF-illumination at basal cell-surface
            # Evanescent field: Amplitude and Depth
            # Assume that the s-polar direction is parallel to y-axis
            A2_x = A2_Ip*(4*cos2*(sin2 - r2)/(r2**2*cos2 + sin2 - r2))
            A2_y = A2_Is*(4*cos2/(1 - r2))
            A2_z = A2_Ip*(4*cos2*sin2/(r2**2*cos2 + sin2 - r2))

            A2_Tp = A2_x + A2_z
            A2_Ts = A2_y

            penetration_depth = wave_length/(4.0*numpy.pi*numpy.sqrt(n_1**2*sin2 - n_2**2))
            penetration_depth = penetration_depth/1e-9

        # set illumination amplitude
        amplitude = (A2_Tp + A2_Ts)/2
        return amplitude, penetration_depth

    def get_coordinate(self, p_i, p_0):
        """Deprecated."""
        return rotate_coordinate(p_i, p_0)

    def polar2cartesian_coordinates(self, r, t, x, y):
        """Deprecated."""
        return polar2cartesian_coordinates(r, t, x, y)

    def polar2cartesian(self, grid, coordinates, shape):
        """Deprecated."""
        return polar2cartesian(grid, coordinates, shape)

    def output_frames(self, dataset, pathto='./images', image_fmt='image_%07d.npy', true_fmt='true_%07d.npy', rng=None):
        # Check and create the folders for image and output files.
        if not os.path.exists(pathto):
            os.makedirs(pathto)

        for frame_index in range(self.num_frames()):
            camera, true_data = self.output_frame(dataset, frame_index, rng=rng)

            # save image to numpy-binary file
            image_file_name = os.path.join(pathto, image_fmt % (frame_index))
            numpy.save(image_file_name, camera)

            # save true-dataset to numpy-binary file
            true_file_name = os.path.join(pathto, true_fmt % (frame_index))
            numpy.save(true_file_name, true_data)

    def output_frame(self, dataset, frame_index, rng=None):
        start_time, end_time, exposure_time = (
            self.configs.shutter_start_time, self.configs.shutter_end_time, self.configs.detector_exposure_time)

        times = numpy.array([t for t, _ in dataset.data])
        t = start_time + exposure_time * frame_index
        assert t + exposure_time <= end_time + 1e-13
        start_index = numpy.searchsorted(times, t, side='right')
        if start_index != 0:
            start_index -= 1
        stop_index = numpy.searchsorted(times, t + exposure_time, side='left')
        if times[start_index] > t:
            warnings.warn("No data input for interval [{}, {}]".format(t, times[start_index]))

        frame_data = dataset.data[start_index: stop_index]

        # set Fluorophores PSF
        # self.set_fluo_psf(dataset, [(frame_index, t, start_index, end_index)])
        fluo_psfs = self.get_fluo_psfs(frame_data, dataset.lengths)

        _log.info('time: {} sec ({})'.format(t, frame_index))

        # cell size (nm scale)
        cell_size = self.get_cell_size(dataset.lengths)
        _, Ny, Nz = cell_size

        # focal point
        p_0 = self.get_focal_center(cell_size)

        # beam position: Assuming beam position = focal point (for temporary)
        p_b = copy.copy(p_0)

        # define cell in nm-scale
        Nw_pixel, Nh_pixel = self.configs.detector_image_size  # in pixels
        pixel_length = self.configs.detector_pixel_length / self.configs.image_magnification / 1e-9  # in nm
        Lw, Lh = Nw_pixel * pixel_length, Nh_pixel * pixel_length  # in nm
        Nw, Nh = math.ceil(Lw), math.ceil(Lh)
        cell = numpy.zeros(shape=(Nw, Nh))

        # loop for frame data
        for i, (i_time, particles) in enumerate(frame_data):
            # When the first i_time is less than the start time given, exposure_time is shorter than expected
            # unit_time = dataset.interval
            current_time = i_time if i != 0 else t
            next_time = frame_data[i + 1][0] if i + 1 < len(frame_data) else t + exposure_time
            unit_time = next_time - current_time

            _log.info('    {:03d}-th file in {:03d}-th frame: {} + {} sec'.format(
                i, frame_index, current_time, unit_time))

            if unit_time < 1e-13:  #XXX: Skip merging when the exposure time is too short
                continue

            # define true-dataset in last-frame
            # [frame-time, m-ID, m-state, p-state, (depth,y0,z0), sqrt(<dr2>)]
            true_data = numpy.zeros(shape=(len(particles), 7))  #XXX: <== true_data just keeps the last state in the frame
            true_data[: , 0] = t

            # loop for particles
            for j, particle_j in enumerate(particles):
                # self.get_molecule_plane(cell, j, particle_j, p_b, p_0, true_data, unit_time, fluo_psfs)
                self.overlay_molecule_plane(cell, j, particle_j, p_b, p_0, true_data, unit_time, fluo_psfs)

        # convert image in pixel-scale
        # camera, true_data = self.detector_output(rng, cell, true_data)
        camera, true_data = self.new_detector_output(rng, cell, true_data, p_0)
        return (camera, true_data)

    def overwrite_smeared(self, cell_pixel, photon_dist, i, j):
        # i-th pixel
        Ni_pixel = len(cell_pixel)
        Ni_pe    = len(photon_dist)

        i_to   = i + Ni_pe/2
        i_from = i - Ni_pe/2

        if (i_to > Ni_pixel):
            di_to = i_to - Ni_pixel
            i10_to = int(Ni_pixel)
            i1_to = int(Ni_pe - di_to)
        else:
            di_to = Ni_pixel - (i + Ni_pe/2)
            i0_to = int(Ni_pixel - di_to)
            i1_to = int(Ni_pe)

        if (i_from < 0):
            di_from = abs(i_from)
            i0_from = 0
            i1_from = int(di_from)
        else:
            di_from = i_from
            i0_from = int(di_from)
            i1_from = 0

        ddi = (i0_to - i0_from) - (i1_to - i1_from)

        if (ddi != 0): i0_to = i0_to - ddi
        if (ddi != 0): i1_to = i1_to - ddi

        # j-th pixel
        Nj_pixel = len(cell_pixel[0])
        Nj_pe    = len(photon_dist[0])

        j_to   = j + Nj_pe/2
        j_from = j - Nj_pe/2

        if (j_to > Nj_pixel):
            dj_to = j_to - Nj_pixel
            j0_to = int(Nj_pixel)
            j1_to = int(Nj_pe - dj_to)
        else:
            dj_to = Nj_pixel - (j + Nj_pe/2)
            j0_to = int(Nj_pixel - dj_to)
            j1_to = int(Nj_pe)

        if (j_from < 0):
            dj_from = abs(j_from)
            j0_from = 0
            j1_from = int(dj_from)
        else:
            dj_from = j_from
            j0_from = int(dj_from)
            j1_from = 0

        ddj = (j0_to - j0_from) - (j1_to - j1_from)

        if (ddj != 0): j0_to = j0_to - ddj
        if (ddj != 0): j1_to = j1_to - ddj

        # add to cellular plane
        cell_pixel[i0_from:i0_to, j0_from:j0_to] += photon_dist[i1_from:i1_to, j1_from:j1_to]

        return cell_pixel

    def prob_EMCCD(self, S, E):
        # get EM gain
        M = self.configs.detector_emgain
        a = 1.00/M

        prob = numpy.zeros(shape=(len(S)))

        if (S[0] > 0):
            prob = numpy.sqrt(a*E/S)*numpy.exp(-a*S-E+2*numpy.sqrt(a*E*S))*i1e(2*numpy.sqrt(a*E*S))
        else:
            prob[0] = numpy.exp(-E)
            prob[1:] = numpy.sqrt(a*E/S[1:])*numpy.exp(-a*S[1:]-E+2*numpy.sqrt(a*E*S[1:]))*i1e(2*numpy.sqrt(a*E*S[1:]))

        return prob

    def new_detector_output(self, rng, cell, true_data, focal_center):
        Nw_pixel, Nh_pixel = self.configs.detector_image_size  # pixels
        pixel_length = self.configs.detector_pixel_length / self.configs.image_magnification / 1e-9  # nm
        _, y0, z0 = focal_center  # nm

        _log.info('scaling [nm/pixel]: {}'.format(pixel_length))
        _log.info('center (width, height): {} {}'.format(z0, y0))

        camera_pixel = numpy.zeros((Nw_pixel, Nh_pixel, 2))

        for i in range(Nw_pixel):
            for j in range(Nh_pixel):
                i0 = max(math.floor(((2 * i - Nw_pixel) * pixel_length + cell.shape[0]) / 2), 0)
                i1 = min(math.floor(((2 * (i + 1) - Nw_pixel) * pixel_length + cell.shape[0]) / 2), cell.shape[0])
                j0 = max(math.floor(((2 * j - Nh_pixel) * pixel_length + cell.shape[1]) / 2), 0)
                j1 = min(math.floor(((2 * (j + 1) - Nh_pixel) * pixel_length + cell.shape[1]) / 2), cell.shape[1])
                if i0 >= i1 or j0 >= j1:
                    continue

                photons = cell[i0: i1, j0: j1].sum()
                if photons <= 0:
                    continue

                # get crosstalk
                if self.effects.crosstalk_switch is True:
                    n_i, n_j = rng.normal(0, self.effects.crosstalk_width, int(photons), 2)

                    smeared_photons, edge_i, edge_j = numpy.histogram2d(
                        n_i, n_j, bins=(24, 24), range=[[-12, 12], [-12, 12]])

                    # smeared photon distributions
                    camera_pixel[: , : , 0] = self.overwrite_smeared(camera_pixel[: , : , 0], smeared_photons, i, j)
                else:
                    camera_pixel[i, j, 0] = photons

        true_data[: , 4] = numpy.floor(
            (numpy.floor(true_data[: , 4] ) - math.floor(z0 - Nw_pixel * pixel_length / 2)) / pixel_length)
        true_data[: , 5] = numpy.floor(
            (numpy.floor(true_data[: , 5] ) - math.floor(y0 - Nh_pixel * pixel_length / 2)) / pixel_length)

        ## CMOS (readout noise probability ditributions)
        if self.configs.detector_type == "CMOS":
            noise_data = numpy.loadtxt(
                os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/detector/RNDist_F40.csv'),
                delimiter=',')
            Nr_cmos = noise_data[: , 0]
            p_noise = noise_data[: , 1]
            p_nsum  = p_noise.sum()

        ## conversion: photon --> photoelectron --> ADC count
        for i in range(Nw_pixel):
            for j in range(Nh_pixel):
                ## Detector: Quantum Efficiency
                # index = int(self.configs.psf_wavelength) - int(self.configs.wave_length[0])
                # QE = self.configs.detector_qeff[index]
                QE = self.configs.detector_qeff

                ## get signal (photons)
                photons = camera_pixel[i, j, 0]

                ## get constant background (photoelectrons)
                photons += self.effects.background_mean

                ## get signal (expectation)
                expected = QE * photons

                ## select Camera type
                if self.configs.detector_type == "CMOS":
                    ## get signal (poisson distributions)
                    signal = rng.poisson(expected, None)

                    ## get detector noise (photoelectrons)
                    noise  = rng.choice(Nr_cmos, None, p=p_noise / p_nsum)
                    Nr = 1.3

                elif self.configs.detector_type == "EMCCD":
                    ## get signal (photoelectrons)
                    if expected > 0:
                        ## get EM gain
                        M = self.configs.detector_emgain

                        ## set probability distributions
                        s_min = max(0, M * int(expected - 5.0 * numpy.sqrt(expected) - 10))
                        s_max = M * int(expected + 5.0 * numpy.sqrt(expected) + 10)
                        # s = numpy.array([k for k in range(s_min, s_max)])
                        s = numpy.arange(s_min, s_max)
                        p_signal = self.prob_EMCCD(s, expected)
                        p_ssum = p_signal.sum()

                        ## get signal (photoelectrons)
                        signal = rng.choice(s, None, p=p_signal / p_ssum)

                    else:
                        signal = 0

                    # get detector noise (photoelectrons)
                    Nr = self.configs.detector_readout_noise
                    noise = rng.normal(0, Nr, None) if Nr > 0 else 0

                elif self.configs.detector_type == "CCD":
                    ## get signal (poisson distributions)
                    signal = rng.poisson(expected, None)

                    ## get detector noise (photoelectrons)
                    Nr = self.configs.detector_readout_noise
                    noise = rng.normal(0, Nr, None) if Nr > 0 else 0

                ## A/D converter: Photoelectrons --> ADC counts
                PE = signal + noise
                ADC = self.get_ADC_value(rng, (i, j), PE)

                # set data in image array
                camera_pixel[i, j, 0] = expected
                camera_pixel[i, j, 1] = ADC

        return camera_pixel, true_data

    def detector_output(self, rng, cell, true_data):
        # Detector Output
        Nw_pixel, Nh_pixel = self.configs.detector_image_size

        Np = self.configs.detector_pixel_length / self.configs.image_magnification / 1e-9

        # cell in nm-scale
        Nw_cell = len(cell)
        Nh_cell = len(cell[0])

        # cell in pixel-scale
        Nw_cell_pixel = int(Nw_cell/Np)
        Nh_cell_pixel = int(Nh_cell/Np)

        # dummy image in pixel-scale
        if (Nw_cell_pixel > Nw_pixel):
            #Nw_dummy = 2*Nw_cell_pixel
            Nw_dummy = Nw_cell_pixel
        else:
            Nw_dummy = 2*Nw_pixel

        if (Nh_cell_pixel > Nh_pixel):
            #Nh_dummy = 2*Nh_cell_pixel
            Nh_dummy = Nh_cell_pixel
        else:
            Nh_dummy = 2*Nh_pixel

        # dummy image in nm-scale
        Nw_camera = Nw_dummy*Np
        Nh_camera = Nh_dummy*Np

        # weidth
        w_cam_from = Nw_camera/2.0 - Nw_cell/2.0
        w_cam_to   = Nw_camera/2.0 + Nw_cell/2.0

        if (w_cam_from < 0):
            w_cel_from = abs(w_cam_from)
            w_cam_from = 0
        else:
            w_cel_from = 0

        if (w_cam_to > Nw_camera):
            w_cel_to = Nw_cell - (w_cam_to - Nw_camera)
            w_cam_to = Nw_camera
        else:
            w_cel_to = Nw_cell

        # height
        h_cam_from = Nh_camera/2.0 - Nh_cell/2.0
        h_cam_to   = Nh_camera/2.0 + Nh_cell/2.0

        if (h_cam_from < 0):
            h_cel_from = abs(h_cam_from)
            h_cam_from = 0
        else:
            h_cel_from = 0

        if (h_cam_to > Nh_camera):
            h_cel_to = Nh_cell - (h_cam_to - Nh_camera)
            h_cam_to = Nh_camera
        else:
            h_cel_to = Nh_cell

        # image in nm-scale
        w_cel0, w_cel1 = int(w_cel_from), int(w_cel_to)
        h_cel0, h_cel1 = int(h_cel_from), int(h_cel_to)

        plane = cell[w_cel0:w_cel1, h_cel0:h_cel1]

        # declear cell image in pixel-scale
        cell_pixel = numpy.zeros([Nw_cell_pixel, Nh_cell_pixel])

        # Signal (photon distribution on cell)
        for i in range(Nw_cell_pixel):
            for j in range(Nh_cell_pixel):

                # get photons
                i0, i1 = int(i*Np), int((i+1)*Np)
                j0, j1 = int(j*Np), int((j+1)*Np)

                photons = numpy.sum(plane[i0:i1,j0:j1])
                #photons = numpy.sum(plane[int(i*Np):int((i+1)*Np),int(j*Np):int((j+1)*Np)])
                #cell_pixel[i][j] = photons

                if (photons > 0):

                    # get crosstalk
                    if (self.effects.crosstalk_switch == True):

                        width = self.effects.crosstalk_width

                        n_i = rng.normal(0, width, int(photons))
                        n_j = rng.normal(0, width, int(photons))

                        smeared_photons, edge_i, edge_j = numpy.histogram2d(n_i, n_j, bins=(24, 24),
                                                                            range=[[-12,12],[-12,12]])

                        # smeared photon distributions
                        cell_pixel = self.overwrite_smeared(cell_pixel, smeared_photons, i, j)

                    else:
                        cell_pixel[i][j] = photons

        # declear photon distribution for dummy image
        dummy_pixel = numpy.zeros([Nw_dummy, Nh_dummy])

        w_cam0 = int(w_cam_from/Np)
        w_cam1 = int(w_cam_to/Np)
        h_cam0 = int(h_cam_from/Np)
        h_cam1 = int(h_cam_to/Np)

        w_cel0 = int(w_cel_from/Np)
        w_cel1 = int(w_cel_to/Np)
        h_cel0 = int(h_cel_from/Np)
        h_cel1 = int(h_cel_to/Np)

        ddw = (w_cam1 - w_cam0) - (w_cel1 - w_cel0)
        ddh = (h_cam1 - h_cam0) - (h_cel1 - h_cel0)

        if (ddw > 0): w_cam1 = w_cam1 - ddw
        if (ddw < 0): w_cel1 = w_cel1 + ddw
        if (ddh > 0): h_cam1 = h_cam1 - ddh
        if (ddh < 0): h_cel1 = h_cel1 + ddh

        # place cell_pixel data to camera image
        dummy_pixel[w_cam0:w_cam1, h_cam0:h_cam1] = cell_pixel[w_cel0:w_cel1, h_cel0:h_cel1]

        # get focal point
        f_x, f_y, f_z = self.configs.detector_focal_point

        # get dummy-image center and focal position
        w0 = Nw_dummy // 2 - int(Nw_cell / Np * (0.5 - f_z))
        h0 = Nh_dummy // 2 - int(Nh_cell / Np * (0.5 - f_y))

        # dummy_pixel image to camera_pixel image
        camera_pixel = numpy.zeros([Nw_pixel, Nh_pixel, 2])
        camera_pixel[:,:,0] = dummy_pixel[w0-Nw_pixel//2:w0+Nw_pixel//2, h0-Nh_pixel//2:h0+Nh_pixel//2]

        _log.info('scaling [nm/pixel]: {}'.format(Np))
        _log.info('center (width, height): {} {}'.format(w0, h0))

        # set true-dataset in pixel-scale
        true_data[:,4] = true_data[:,4]/Np - w_cel0 + w_cam0 - (w0 - (Nw_pixel-1)/2.0)
        true_data[:,5] = true_data[:,5]/Np - h_cel0 + h_cam0 - (h0 - (Nh_pixel-1)/2.0)

        # CMOS (readout noise probability ditributions)
        if (self.configs.detector_type == "CMOS"):
            noise_data = numpy.loadtxt(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                    'catalog/detector/RNDist_F40.csv'), delimiter=',')
            Nr_cmos = noise_data[:,0]
            p_noise = noise_data[:,1]
            p_nsum  = p_noise.sum()

        # conversion: photon --> photoelectron --> ADC count
        for i in range(Nw_pixel):
            for j in range(Nh_pixel):
                # pixel position
                pixel = (i, j)

                # Detector: Quantum Efficiency
                #index = int(self.configs.psf_wavelength) - int(self.configs.wave_length[0])
                #QE = self.configs.detector_qeff[index]
                QE = self.configs.detector_qeff

                # get signal (photons)
                Photons = camera_pixel[i][j][0]

                # get constant background (photoelectrons)
                Photons_bg = self.effects.background_mean
                Photons += Photons_bg

                # get signal (expectation)
                Exp = QE*Photons

                # select Camera type
                if (self.configs.detector_type == "CMOS"):

                    # get signal (poisson distributions)
                    signal = rng.poisson(Exp, None)

                    # get detector noise (photoelectrons)
                    noise  = rng.choice(Nr_cmos, None, p=p_noise/p_nsum)
                    Nr = 1.3


                elif (self.configs.detector_type == "EMCCD"):

                    # get signal (photoelectrons)
                    if (Exp > 0):

                        # get EM gain
                        M = self.configs.detector_emgain

                        # set probability distributions
                        s_min = M*int(Exp - 5.0*numpy.sqrt(Exp) - 10)
                        s_max = M*int(Exp + 5.0*numpy.sqrt(Exp) + 10)

                        if (s_min < 0): s_min = 0

                        s = numpy.array([k for k in range(s_min, s_max)])
                        p_signal = self.prob_EMCCD(s, Exp)
                        p_ssum = p_signal.sum()

                        # get signal (photoelectrons)
                        signal = rng.choice(s, None, p=p_signal/p_ssum)

                    else:
                        signal = 0

                    # get detector noise (photoelectrons)
                    Nr = self.configs.detector_readout_noise

                    if (Nr > 0):
                        noise = rng.normal(0, Nr, None)
                    else: noise = 0


                elif (self.configs.detector_type == "CCD"):

                    # get signal (poisson distributions)
                    signal = rng.poisson(Exp, None)

                    # get detector noise (photoelectrons)
                    Nr = self.configs.detector_readout_noise

                    if (Nr > 0):
                        noise = rng.normal(0, Nr, None)
                    else: noise = 0

                # A/D converter: Photoelectrons --> ADC counts
                PE  = signal + noise
                ADC = self.get_ADC_value(rng, pixel, PE)

                # set data in image array
                camera_pixel[i][j] = [Exp, ADC]

        return camera_pixel, true_data

    def get_ADC_value(self, rng, pixel, photoelectron):
        # pixel position
        i, j = pixel

        # check non-linearity
        fullwell = self.configs.ADConverter_fullwell

        if (photoelectron > fullwell):
            photoelectron = fullwell

        # convert photoelectron to ADC counts
        k = self.configs.ADConverter_gain[i][j]
        ADC0 = self.configs.ADConverter_offset[i][j]
        ADC_max = 2**self.configs.ADConverter_bit - 1

        ADC = photoelectron/k + ADC0

        if (ADC > ADC_max): ADC = ADC_max
        if (ADC < 0): ADC = 0

        #return int(ADC)
        return ADC

    def set_fluo_psf(self, dataset, frames):
        data = []
        for frame_index, t, start_index, stop_index in frames:
            data.extend(dataset.data[start_index: stop_index])

        self.fluo_psf = self.get_fluo_psfs(data, dataset.lengths)

    def get_fluo_psfs(self, data, lengths):
        _log.info("get_fluo_psfs was called.")

        # get cell size
        Nx, Ny, Nz = self.get_cell_size(lengths)

        # focal point
        f_0 = self.configs.detector_focal_point
        p_0 = numpy.array([Nx, Ny, Nz]) * f_0

        depths = []
        for _, particles in data:
            for particle in particles:
                coordinate = particle[0]
                p_i = numpy.array(coordinate) / 1e-9

                # Snell's law
                # amplitude, penet_depth = self.snells_law(p_i, p_0)

                # Particle coordinte in real(nm) scale
                p_i, radial, depth = rotate_coordinate(p_i, p_0)

                # fluorophore axial position
                fluo_depth = int(depth) if depth < len(self.configs.depth) else -1

                depths.append(fluo_depth)

        depths = list(set(depths))

        r = self.configs.radial

        theta = numpy.linspace(0, 90, 91)
        z = numpy.linspace(0, +r[-1], len(r))
        y = numpy.linspace(0, +r[-1], len(r))

        coordinates = polar2cartesian_coordinates(r, theta, z, y)
        psf_t = numpy.ones_like(theta)

        fluo_psfs = {}
        for depth in depths:
            psf_r = self.configs.fluorophore_psf[depth]
            psf_polar = numpy.array(list(map(lambda x: psf_t * x, psf_r)))
            fluo_psfs[depth] = polar2cartesian(psf_polar, coordinates, (len(r), len(r)))

        return fluo_psfs

    def get_nprocs(self):
        return self.__nprocs

    def use_multiprocess(self):
        """Deprecated."""
        warnings.warn('This is no longer supported.')
        return (self.get_nprocs() != 1)
        # envname = 'ECELL_MICROSCOPE_SINGLE_PROCESS'
        # return not (envname in os.environ and os.environ[envname])
