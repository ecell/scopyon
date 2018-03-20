import sys
import os
import shutil
import copy
import tempfile
import math
import operator
# import h5py
import csv
import string
import ctypes
import multiprocessing
import warnings

from ast import literal_eval

import numpy

import scipy
from scipy.special import j0, i1, i1e
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
#from scipy.misc    import toimage
#from PIL.Image import fromarray

from . import parameter_configs
from . import parameter_effects
from .effects import PhysicalEffects
from . import io

 # import numpy.random as rng

from logging import getLogger
_log = getLogger(__name__)


class VisualizerError(Exception):
    "Exception class for visualizer"

    def __init__(self, info):
        self.__info = info

    def __repr__(self):
        return self.__info

    def __str__(self):
        return self.__info

class EPIFMConfigs():
    '''
    EPIFM Configuration

        Wide-field Gaussian Profile
            +
        Detector: EMCCD/CMOS
    '''

    def __init__(self, user_configs_dict = None):
        # default setting
        configs_dict = parameter_configs.__dict__.copy()

        # user setting
        if user_configs_dict is not None:
            if type(user_configs_dict) != type({}):
                _log.info('Illegal argument type for constructor of Configs class')
                sys.exit()
            configs_dict.update(user_configs_dict)

        for key, val in configs_dict.items():
            if key[0] != '_': # Data skip for private variables in setting_dict.
                if type(val) == type({}) or type(val) == type([]):
                    copy_val = copy.deepcopy(val)
                else:
                    copy_val = val
                setattr(self, key, copy_val)

    def _set_data(self, key, val):
        if val is not None:
            setattr(self, key, val)

    def set_shutter(self, start_time = None,
                          end_time   = None,
                          time_open  = None,
                          time_lapse = None):
        self._set_data('shutter_switch', True)
        self._set_data('shutter_start_time', start_time)
        self._set_data('shutter_end_time', end_time)

        if (time_open is None or time_lapse is None):
            time_open  = end_time - start_time
            time_lapse = end_time - start_time

        self._set_data('shutter_time_open', time_open)
        self._set_data('shutter_time_lapse', time_lapse)

        _log.info('--- Shutter:')
        _log.info('    Start-Time = {} sec'.format(self.shutter_start_time))
        _log.info('    End-Time   = {} sec'.format(self.shutter_end_time))
        _log.info('    Time-open  = {} sec'.format(self.shutter_time_open))
        _log.info('    Time-lapse = {} sec'.format(self.shutter_time_lapse))

    def set_light_source(self, source_type = None,
                               wave_length = None,
                               flux_density = None,
                               center = None,
                               radius = None,
                               angle  = None):
        self._set_data('source_switch', True)
        self._set_data('source_type', source_type)
        self._set_data('source_wavelength', wave_length)
        self._set_data('source_flux_density', flux_density)
        self._set_data('source_radius', radius)
        self._set_data('source_angle', angle)

        _log.info('--- Light Source:{}'.format(self.source_type))
        _log.info('    Wave Length = {} nm'.format(self.source_wavelength))
        _log.info('    Beam Flux Density = {} W/cm2'.format(self.source_flux_density))
        _log.info('    1/e2 Radius = {} m'.format(self.source_radius))
        _log.info('    Angle = {} degree'.format(self.source_angle))

    def set_ExcitationFilter(self, excitation = None):
        _log.info('--- Excitation Filter:')
        filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/excitation/') + excitation + '.csv'

        try:
            csvfile = open(filename)
            lines = csvfile.readlines()

            header = lines[0:5]
            data   = lines[6:]

            excitation_header = []
            excitation_filter = []

            for i in range(len(header)):
                dummy  = header[i].split('\r\n')
                a_data = dummy[0].split(',')
                excitation_header.append(a_data)
                _log.info('    {}'.format(a_data))

            for i in range(len(data)):
                dummy0 = data[i].split('\r\n')
                a_data = dummy0[0].split(',')
                excitation_filter.append(a_data)

        except Exception:
            _log.error('Error: {} is NOT found'.format(filename))
            exit()

        ####
        self.excitation_eff = self.set_efficiency(excitation_filter)
        self._set_data('excitation_switch', True)

    def set_fluorophore(self, fluorophore_type = None,
                              wave_length = None,
                              normalization = None,
                              width = None,
                              cutoff = None,
                              file_name_format = None ):
        if (fluorophore_type == 'Gaussian'):
            _log.info('--- Fluorophore: %s PSF' % (fluorophore_type))

            self._set_data('fluorophore_type {}'.format(fluorophore_type))
            self._set_data('psf_wavelength {}'.format(wave_length))
            self._set_data('psf_normalization {}'.format(normalization))
            self._set_data('psf_width {}'.format(width))
            self._set_data('psf_cutoff {}'.format(cutoff))
            self._set_data('psf_file_name_format {}'.format(file_name_format))

            index = (numpy.abs(self.wave_length - self.psf_wavelength)).argmin()

            self.fluoex_eff[index] = 100
            self.fluoem_eff[index] = 100

            _log.info('    Wave Length   =  {} nm'.format(self.psf_wavelength))
            _log.info('    Normalization =  {}'.format(self.psf_normalization))
            _log.info('    Lateral Width =  {} nm'.format(self.psf_width[0]))
            _log.info('    Axial Width =  {} nm'.format(self.psf_width[1]))
            #print '    Lateral Cutoff = ', self.psf_cutoff[0], 'nm'
            #print '    Axial Cutoff = ', self.psf_cutoff[1], 'nm'

        else:
            _log.info('--- Fluorophore:')
            filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/fluorophore/', fluorophore_type + '.csv')

            if not os.path.exists(filename):
                _log.info('Error: ', filename, ' is NOT found')
                exit()

            with open(filename) as csvfile:
                lines = [_.rstrip() for _ in csvfile.readlines()]

                header = lines[0:5]
                data   = lines[5:]

                fluorophore_header     = [_.split(',') for _ in header]
                for _ in fluorophore_header: _log.info("     {}".format(_))

                em_idx = data.index('Emission')
                fluorophore_excitation = [_.split(',') for _ in data[1:em_idx]]
                fluorophore_emission   = [_.split(',') for _ in data[(em_idx + 1):]]

            ####
            self.fluoex_eff = self.set_efficiency(fluorophore_excitation)
            self.fluoem_eff = self.set_efficiency(fluorophore_emission)

            index_ex = self.fluoex_eff.index(max(self.fluoex_eff))
            index_em = self.fluoem_eff.index(max(self.fluoem_eff))

            #### for temporary
            self._set_data('fluorophore_type', fluorophore_type)
            self._set_data('psf_wavelength', self.wave_length[index_em])
            self._set_data('psf_normalization', normalization)
            self._set_data('psf_file_name_format', file_name_format)

            self.fluoex_eff[index_ex] = 100
            self.fluoem_eff[index_em] = 100

            _log.info('    PSF Normalization Factor =  {}'.format(self.psf_normalization))
            _log.info('    Excitation: Wave Length =  {} nm'.format(self.wave_length[index_ex]))
            _log.info('    Emission  : Wave Length =  {} nm'.format(self.psf_wavelength))

        # Normalization
        norm = sum(self.fluoex_eff)
        self.fluoex_norm = numpy.array(self.fluoex_eff)/norm

        norm = sum(self.fluoem_eff)
        self.fluoem_norm = numpy.array(self.fluoem_eff)/norm

    def set_dichroic_mirror(self, dm = None):
        _log.info('--- Dichroic Mirror:')
        filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/dichroic/') + dm + '.csv'

        try:
            csvfile = open(filename)
            lines = csvfile.readlines()

            header = lines[0:5]
            data   = lines[6:]

            dichroic_header = []
            dichroic_mirror = []

            for i in range(len(header)):
                dummy  = header[i].split('\r\n')
                a_data = dummy[0].split(',')
                dichroic_header.append(a_data)
                _log.info('     {}'.format(a_data))

            for i in range(len(data)):
                dummy0 = data[i].split('\r\n')
                a_data = dummy0[0].split(',')
                dichroic_mirror.append(a_data)

        except Exception:
            _log.error('Error: {} is NOT found'.format(filename))
            exit()

        self.dichroic_eff = self.set_efficiency(dichroic_mirror)
        self._set_data('dichroic_switch', True)

    def set_EmissionFilter(self, emission = None):
        _log.info('--- Emission Filter:')
        filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/emission/') + emission + '.csv'

        try:
            csvfile = open(filename)
            lines = csvfile.readlines()

            header = lines[0:5]
            data   = lines[6:]

            emission_header = []
            emission_filter = []

            for i in range(len(header)):
                dummy  = header[i].split('\r\n')
                a_data = dummy[0].split(',')
                emission_header.append(a_data)
                _log.info('    {}'.format(a_data))

            for i in range(len(data)):
                dummy0 = data[i].split('\r\n')
                a_data = dummy0[0].split(',')
                emission_filter.append(a_data)

        except Exception:
            _log.info('Error: ', filename, ' is NOT found')
            exit()

        self.emission_eff = self.set_efficiency(emission_filter)
        self._set_data('emission_switch', True)

    def set_magnification(self, magnification = None):
        self._set_data('image_magnification', magnification)
        _log.info('--- Magnification: x {}'.format(self.image_magnification))

    def set_detector(self, detector = None,
                   image_size = None,
                   pixel_length = None,
                   exposure_time = None,
                   focal_point = None,
                   QE = None,
                   readout_noise = None,
                   dark_count = None,
                   emgain = None
                   ):
        self._set_data('detector_switch', True)
        self._set_data('detector_type', detector)
        self._set_data('detector_image_size', image_size)
        self._set_data('detector_pixel_length', pixel_length)
        self._set_data('detector_focal_point', focal_point)
        #self._set_data('detector_base_position', base_position)
        self._set_data('detector_exposure_time', exposure_time)
        self._set_data('detector_qeff', QE)
        self._set_data('detector_readout_noise', readout_noise)
        self._set_data('detector_dark_count', dark_count)
        self._set_data('detector_emgain', emgain)

        _log.info('--- Detector:  {}'.format(self.detector_type))
        _log.info('    Image Size  =  {} x {}'.format(self.detector_image_size[0], self.detector_image_size[1]))
        _log.info('    Pixel Size  =  {} m/pixel'.format(self.detector_pixel_length))
        _log.info('    Focal Point =  {}'.format(self.detector_focal_point))
        #print '    Position    =  {}'.format(self.detector_base_position)
        _log.info('    Exposure Time =  {} sec'.format(self.detector_exposure_time))
        _log.info('    Quantum Efficiency =  {} %'.format(100*self.detector_qeff))
        _log.info('    Readout Noise =  {} electron'.format(self.detector_readout_noise))
        _log.info('    Dark Count =  {} electron/sec'.format(self.detector_dark_count))
        _log.info('    EM gain = x {}'.format(self.detector_emgain))

    def set_analog_to_digital_converter(self, bit = None,
                        gain = None,
                        offset = None,
                        fullwell = None,
                        fpn_type = None,
                        fpn_count = None,
                        rng = None):
        self._set_data('ADConverter_bit', bit)
        self._set_data('ADConverter_gain', (fullwell - 0.0)/(pow(2.0, bit) - offset))
        self._set_data('ADConverter_offset', offset)
        self._set_data('ADConverter_fullwell', fullwell)
        self._set_data('ADConverter_fpn_type', fpn_type)
        self._set_data('ADConverter_fpn_count', fpn_count)

        _log.info('--- A/D Converter: %d-bit' % (self.ADConverter_bit))
        _log.info('    Gain = %.3f electron/count' % (self.ADConverter_gain))
        _log.info('    Offset =  {} count'.format(self.ADConverter_offset))
        _log.info('    Fullwell =  {} electron'.format(self.ADConverter_fullwell))
        _log.info('    {}-Fixed Pattern Noise: {} count'.format(self.ADConverter_fpn_type, self.ADConverter_fpn_count))

        self.set_analog_to_digital_converter_gain(rng)

    def set_analog_to_digital_converter_gain(self, rng):
        # image pixel-size
        Nw_pixel = self.detector_image_size[0]
        Nh_pixel = self.detector_image_size[1]

        # set ADC parameters
        bit  = self.ADConverter_bit
        fullwell = self.ADConverter_fullwell
        ADC0 = self.ADConverter_offset

        # set Fixed-Pattern noise
        FPN_type  = self.ADConverter_fpn_type
        FPN_count = self.ADConverter_fpn_count

        if (FPN_type is None):
            # offset = numpy.empty(Nw_pixel*Nh_pixel)
            # offset.fill(ADC0)
            offset = numpy.array([ADC0 for i in range(Nw_pixel * Nh_pixel)])

        elif (FPN_type == 'pixel'):
            if rng is None:
                raise RuntimeError('A random number generator is required.')
            offset = numpy.rint(rng.normal(ADC0, FPN_count, Nw_pixel * Nh_pixel))

        elif (FPN_type == 'column'):
            column = rng.normal(ADC0, FPN_count, Nh_pixel)
            temporal = numpy.array([column for i in range(Nw_pixel)])
            offset = numpy.rint(temporal.reshape(Nh_pixel*Nw_pixel))

        # set ADC gain
        # gain = numpy.array(map(lambda x: (fullwell - 0.0)/(pow(2.0, bit) - x), offset))
        gain = (fullwell - 0.0) / ( pow(2.0, bit) - offset)

        # reshape
        self.ADConverter_offset = offset.reshape([Nw_pixel, Nh_pixel])
        self.ADConverter_gain = gain.reshape([Nw_pixel, Nh_pixel])

    def set_efficiency(self, array, index=1):
        array = numpy.array(array, dtype = 'float')
        array = array[array[:, 0] % 1 == 0,:]

        efficiency = numpy.zeros(len(self.wave_length))
        idx1 = numpy.in1d(numpy.array(self.wave_length), array[:, 0])
        idx2 = numpy.in1d(numpy.array(array[:, 0]), self.wave_length)

        efficiency[idx1] = array[idx2, 1]

        return efficiency.tolist()

    # def set_optical_path(self):
    #     # (0) Data: Cell Model Sample
    #     self.set_Time_arrays()
    #     # self.set_Spatiocyte_data_arrays(csv_file_directory, max_count)

    #     # (1) Illumination path: Light source --> Cell Model Sample
    #     # self.set_Illumination_path()
    #     # exit()

    #     # (2) Detection path: Cell Model Sample --> Detector
    #     self.set_Detection_path()

    # def set_Time_arrays(self):
    #     # set time-arrays
    #     start = self.shutter_start_time
    #     end = self.shutter_end_time

    #     # set count arrays by spatiocyte interval
    #     interval = self.spatiocyte_interval
    #     N_count = int(round((end - start)/interval))
    #     c0 = int(round(start/interval))

    #     delta_array = numpy.zeros(shape=(N_count))
    #     delta_array.fill(interval)
    #     time_array  = numpy.cumsum(delta_array) + start
    #     count_array = numpy.array([c + c0 for c in range(N_count)])

    #     # set index arrays by exposure time
    #     exposure = self.detector_exposure_time
    #     N_index = int(round((end - start)/exposure))
    #     i0 = int(round(start/exposure))
    #     index_array = numpy.array([i + i0 for i in range(N_index)])

    #     # set time, count and delta arrays
    #     # self._set_data('shutter_time_array', time_array.tolist())
    #     # self._set_data('shutter_delta_array', delta_array.tolist())
    #     # self._set_data('shutter_count_array', count_array)
    #     # self._set_data('shutter_index_array', index_array)
    #     # self._set_data('shutter_index_array_size', len(index_array))
    #     # self._set_data('shutter_index_array_first', index_array[0])
    #     return (count_array, len(index_array), index_array[0])

    def set_illumination_path(self, detector_focal_point, detector_focal_norm):
        self.detector_focal_point = detector_focal_point
        self.detector_focal_norm = detector_focal_norm
        _log.info('Focal Center: {}'.format(self.detector_focal_point))
        _log.info('Normal Vector: {}'.format(self.detector_focal_norm))

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

    def set_PSF_detector(self):
        r = self.radial
        z = self.depth

        wave_length = self.psf_wavelength

        # Fluorophores Emission Intensity (wave_length)
        I = self.fluoem_norm

        # Photon Transmission Efficiency
        if (self.dichroic_switch == True):
            I = I*0.01*self.dichroic_eff

        if (self.emission_switch == True):
            I = I*0.01*self.emission_eff

        # For normalization
        norm = list(map(lambda x: True if x > 1e-4 else False, I))

        # PSF: Fluorophore
        psf_fl = None

        if (self.fluorophore_type == 'Gaussian'):
            N0 = self.psf_normalization
            Ir = sum(list(map(lambda x: x*numpy.exp(-0.5*(r/self.psf_width[0])**2), norm)))
            Iz = sum(list(map(lambda x: x*numpy.exp(-0.5*(z/self.psf_width[1])**2), norm)))

            psf_fl = numpy.sum(I)*N0*numpy.array(list(map(lambda x: Ir*x, Iz)))
        else:
            # make the norm and wave_length array shorter
            psf_fl = numpy.sum(I)*self.get_PSF_fluorophore(r, z, wave_length)

        self.fluorophore_psf = psf_fl

    def get_PSF_fluorophore(self, r, z, wave_length):
        # set Magnification of optical system
        M = self.image_magnification

        # set Numerical Appature
        NA = 1.4#self.objective_NA

        # set alpha and gamma consts
        k = 2.0*numpy.pi/wave_length
        alpha = k*NA
        gamma = k*(NA/2)**2

        # set rho parameters
        N = 100
        drho = 1.0/N
        rho = numpy.array([(i+1)*drho for i in range(N)])

        J0 = numpy.array(list(map(lambda x: j0(x*alpha*rho), r)))
        Y  = numpy.array(list(map(lambda x: numpy.exp(-2*1.j*x*gamma*rho**2)*rho*drho, z)))
        I  = numpy.array(list(map(lambda x: x*J0, Y)))
        I_sum = I.sum(axis=2)

        # set normalization factor
        Norm = self.psf_normalization

        # set PSF
        psf = Norm*numpy.array(list(map(lambda x: abs(x)**2, I_sum)))
        return psf

    def set_Shutter(self, start_time = None,
                        end_time   = None,
                        time_open  = None,
                        time_lapse = None):
        """Deprecated"""
        warnings.warn('This is no longer supported.')
        self.set_shutter(start_time, end_time, time_open, time_lapse)

    def set_LightSource(self, source_type = None,
                              wave_length = None,
                              flux_density = None,
                              center = None,
                              radius = None,
                              angle  = None):
        """Deprecated"""
        warnings.warn('This is no longer supported.')
        self.set_light_source(source_type,
                             wave_length,
                             flux_density,
                             center,
                             radius,
                             angle)

    def set_Fluorophore(self, fluorophore_type = None,
                              wave_length = None,
                              normalization = None,
                              width = None,
                              cutoff = None,
                              file_name_format = None ):
        """Deprecated"""
        warnings.warn('This is no longer supported.')
        self.set_fluorophore(fluorophore_type,
                              wave_length,
                              normalization,
                              width,
                              cutoff,
                              file_name_format)

    def set_DichroicMirror(self, dm = None):
        """Deprecated"""
        warnings.warn('This is no longer supported.')
        self.set_dichroic_mirror(dm)

    def set_Magnification(self, Mag = None):
        """Deprecated"""
        warnings.warn('This is no longer supported.')
        self.set_magnification(Mag)

    def set_Detector(self, detector = None,
                   image_size = None,
                   pixel_length = None,
                   exposure_time = None,
                   focal_point = None,
                   QE = None,
                   readout_noise = None,
                   dark_count = None,
                   emgain = None
                   ):
        """Deprecated"""
        warnings.warn('This is no longer supported.')
        self.set_detector(detector,
                       image_size,
                       pixel_length,
                       exposure_time,
                       focal_point,
                       QE,
                       readout_noise,
                       dark_count,
                       emgain)

    def set_ADConverter(self, bit = None,
                        gain = None,
                        offset = None,
                        fullwell = None,
                        fpn_type = None,
                        fpn_count = None):
        """Deprecated"""
        warnings.warn('This is no longer supported.')
        import numpy.random
        rng = numpy.random
        self.set_analog_to_digital_converter(
                        rng,
                        bit,
                        gain,
                        offset,
                        fullwell,
                        fpn_type,
                        fpn_count)

    def set_OutputData(self, image_file_dir = None,
                        image_file_name_format = None,
                        image_file_cleanup_dir=False):
        """Deprecated"""
        warnings.warn('This is no longer supported.')
        self.set_output_path(image_file_dir,
                        image_file_name_format,
                        image_file_cleanup_dir)

    def set_ShapeFile(self, csv_file_directory):
        """Deprecated. Use load_shape instead."""
        warnings.warn('This is no longer supported.')
        self.load_shape(os.path.join(csv_file_directory, 'pt-shape.csv'))

    def set_InputFile(self, csv_file_directory, observable=None):
        """Deprecated. Use load_shape instead."""
        warnings.warn('This is no longer supported.')
        self.load_input(os.path.join(csv_file_directory, 'pt-input.csv'), observable)

    # def set_Optical_path(self, csv_file_directory):
    #     """Deprecated. Use load_shape instead."""
    #     warnings.warn('This is no longer supported.')
    #     self.set_optical_path(csv_file_directory)

class EPIFMVisualizer:
    '''
    EPIFM Visualization class of e-cell simulator
    '''

    def __init__(self, configs=EPIFMConfigs(), effects=PhysicalEffects(), nprocs=1):
        assert isinstance(configs, EPIFMConfigs)
        self.configs = configs

        assert isinstance(effects, PhysicalEffects)
        self.effects = effects

        self.__nprocs = nprocs

    def get_cell_size(self, voxel_radius, lengths):
        # define observational image plane in nm-scale
        voxel_size = 2 * voxel_radius / 1e-9
        # voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9

        ## cell size (nm scale)
        Nz = int(lengths[2] * voxel_size)
        Ny = int(lengths[1] * voxel_size)
        Nx = int(lengths[0] * voxel_size)
        # Nz = int(self.configs.spatiocyte_lengths[2]*voxel_size)
        # Ny = int(self.configs.spatiocyte_lengths[1]*voxel_size)
        # Nx = int(self.configs.spatiocyte_lengths[0]*voxel_size)

        return numpy.array([Nx, Ny, Nz])

    def get_focal_center(self, voxel_radius, lengths):
        # get cell size
        cell_size = self.get_cell_size(voxel_radius, lengths)

        # focal point
        p_0 = cell_size * self.configs.detector_focal_point

        return p_0

    def rewrite_input_data(self, dataset, N_particles=4117, rng=None):
        # get focal point
        p_0 = self.get_focal_center(dataset.voxel_radius, dataset.lengths)

        # beam position: Assuming beam position = focal point (for temporary)
        p_b = copy.copy(p_0)

        # Snell's law
        amplitude0, penet_depth = self.snells_law(p_0, p_0)

        # get the number of emitted photons
        N_emit0 = self.get_emit_photons(amplitude0, dataset.interval, dataset.voxel_radius)

        # # copy shape-file
        # csv_shape = self.configs.spatiocyte_file_directory + '/pt-shape.csv'
        # shutil.copyfile(csv_shape, output_file_dir + '/pt-shape.csv')

        # # get the total number of particles
        # N_particles = 4117 #len(csv_list)  #XXX: Oops!!! Hard-coded here!!!

        # set molecule-states
        molecule_states = numpy.zeros(shape=(N_particles))

        # set fluorescence
        delta_array = numpy.full(shape=(dataset.N_count), fill_value=dataset.interval)
        fluorescence_state, fluorescence_budget = self.effects.get_photophysics_for_epifm(delta_array, N_emit0, N_particles, rng)

        new_data = []
        for k in range(len(dataset.data)):
            # read input file
            # csv_file_path = self.configs.spatiocyte_file_directory + '/pt-%09d.0.csv' % (count_array[k])
            # csv_list = list(csv.reader(open(csv_file_path, 'r')))
            # dataset = numpy.array(csv_list)
            t, particles = dataset.data[k]

            # set molecular-states (unbound-bound)
            # self.set_molecular_states(k, dataset)
            self.initialize_molecular_states(molecule_states, k, particles)

            # set photobleaching-dataset arrays
            self.update_fluorescence_photobleaching(fluorescence_state, fluorescence_budget, k, particles, dataset.voxel_radius, dataset.lengths, dataset.interval)

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

        return new_data

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

    # def set_molecular_states(self, count, dataset):
    #     # reset molecule-states
    #     self.molecule_states.fill(0)

    #     # loop for particles
    #     for j, data_j in enumerate(dataset):

    #         # set particle position
    #         p_i = numpy.array(data_j[1:4]).astype('float')/1e-9

    #         # Molecule ID and its state
    #         m_id, s_id = literal_eval(data_j[5])
    #         # Fluorophore ID and compartment ID
    #         f_id, l_id = literal_eval(data_j[6])

    #         # set molecule-states
    #         self.molecule_states[m_id] = int(s_id)

    def update_fluorescence_photobleaching(self, fluorescence_state, fluorescence_budget, count, data, voxel_radius, lengths, interval):
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
            #XXX:                             args=(count, data[start_index:stop_index], voxel_radius, lengths, c))
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
            state_pb, budget = self.get_fluorescence_photobleaching(fluorescence_state, fluorescence_budget, count, data, voxel_radius, lengths, interval)

        # reset global-arrays for photobleaching-state and photon-budget
        for key, value in state_pb.items():
            fluorescence_state[key, count] = state_pb[key]
            fluorescence_budget[key] = budget[key]
            # self.effects.fluorescence_state[key,count] = state_pb[key]
            # self.effects.fluorescence_budget[key] = budget[key]

    def get_fluorescence_photobleaching(self, fluorescence_state, fluorescence_budget, count, data, voxel_radius, lengths, interval):
        # get focal point
        p_0 = self.get_focal_center(voxel_radius, lengths)

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
            p_i, radial, depth = self.get_coordinate(p_i, p_0)

            state_j = 1  # particles given is always observable. already filtered when read

            # get exponential amplitude (only for observation at basal surface)
            # amplitide = amplitude * numpy.exp(-depth / pent_depth)

            # get the number of emitted photons
            N_emit = self.get_emit_photons(amplitude, interval, voxel_radius)

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

    def get_molecule_plane(self, cell, j, particle_j, p_b, p_0, true_data, dataset):
        # particles coordinate, species and lattice-IDs
        c_id, m_id, s_id, l_id, p_state, cyc_id = particle_j

        # # check if the particle (species) is observable or not
        # sid_array = numpy.array(dataset.species_id)
        # sid_index = (numpy.abs(sid_array - int(s_id))).argmin()
        # if dataset.observables[sid_index] is not True:
        #     return

        #if (p_state > 0):

        p_i = numpy.array(c_id)/1e-9

        # Snell's law
        amplitude, penet_depth = self.snells_law(p_i, p_0)

        # particles coordinate in real(nm) scale
        p_i, radial, depth = self.get_coordinate(p_i, p_0)

        # get exponential amplitude (only for TIRFM-configuration)
        amplitude = amplitude*numpy.exp(-depth/penet_depth)

        # get signal matrix
        signal = self.get_signal(amplitude, radial, depth, p_state, dataset.interval, dataset.voxel_radius)

        # add signal matrix to image plane
        self.overwrite_signal(cell, signal, p_i)

        # set true-dataset
        true_data[j,1] = m_id # molecule-ID
        true_data[j,2] = int(s_id) # sid_index # molecular-state
        true_data[j,3] = p_state # photon-state
        true_data[j,4] = p_i[2] # Y-coordinate in the image-plane
        true_data[j,5] = p_i[1] # X-coordinate in the image-plane
        true_data[j,6] = depth  # Depth from focal-plane

    def get_signal(self, amplitude, radial, depth, p_state, unit_time, voxel_radius):
        # fluorophore axial position
        fluo_depth = depth if depth < len(self.configs.depth) else -1

        # get fluorophore PSF
        psf_depth = self.fluo_psf[int(fluo_depth)]

        # get the number of photons emitted
        N_emit = self.get_emit_photons(amplitude, unit_time, voxel_radius)

        # get signal
        signal = p_state*N_emit/(4.0*numpy.pi)*psf_depth

        return signal

    def get_emit_photons(self, amplitude, unit_time, voxel_radius):
        # Spatiocyte time interval [sec]
        # unit_time = self.configs.spatiocyte_interval

        # Absorption coeff [1/(cm M)]
        abs_coeff = self.effects.abs_coefficient

        # Quantum yield
        QY = self.effects.quantum_yield

        # Abogadoro's number
        Na = self.effects.avogadoros_number

        # Cross-section [m2]
        x_sec = numpy.log(10)*abs_coeff*0.1/Na

        # get the number of absorption photons: [#/(m2 sec)]*[m2]*[sec]
        n_abs = amplitude*x_sec*unit_time

        # spatiocyte voxel size (~ molecule size)
        # voxel_radius = self.configs.spatiocyte_VoxelRadius
        voxel_volume = (4.0/3.0)*numpy.pi*voxel_radius**3
        voxel_depth  = 2.0*voxel_radius

        # Beer-Lamberts law: A = log(I0/I) = abs coef. * concentration * path length ([m2]*[#/m3]*[m])
        A = (abs_coeff*0.1/Na)*(1.0/voxel_volume)*(voxel_depth)

        # get the number of photons emitted
        n_emit = QY*n_abs*(1 - 10**(-A))

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
        x_0, y_0, z_0 = p_0
        x_i, y_i, z_i = p_i

        # Rotation of focal plane
        cos_th0 = 1
        sin_th0 = numpy.sqrt(1 - cos_th0**2)

        # Rotational matrix along z-axis
        #Rot = numpy.matrix([[cos_th, -sin_th, 0], [sin_th, cos_th, 0], [0, 0, 1]])
        Rot = numpy.matrix([[cos_th0, -sin_th0, 0], [sin_th0, cos_th0, 0], [0, 0, 1]])

        # Vector of focal point to particle position
        vec = p_i - p_0
        len_vec = numpy.sqrt(numpy.sum(vec**2))

        # Rotated particle position
        v_rot = Rot*vec.reshape((3,1))
        p_i = numpy.array(v_rot).ravel() + p_0

        # Normal vector of the focal plane
        q_0 = numpy.array([0.0, y_0, 0.0])
        q_1 = numpy.array([0.0, 0.0, z_0])
        R_q0 = numpy.sqrt(numpy.sum(q_0**2))
        R_q1 = numpy.sqrt(numpy.sum(q_1**2))

        q0_rot = Rot*q_0.reshape((3,1))
        q1_rot = Rot*q_1.reshape((3,1))

        norm_v = numpy.cross(q0_rot.ravel(), q1_rot.ravel())/(R_q0*R_q1)

        # Radial distance and depth to focal plane
        cos_0i = numpy.sum(norm_v*vec)/(1.0*len_vec)
        sin_0i = numpy.sqrt(1 - cos_0i**2)

        focal_depth  = abs(len_vec*cos_0i)
        focal_radial = abs(len_vec*sin_0i)

        return p_i, focal_radial, focal_depth

    def polar2cartesian_coordinates(self, r, t, x, y):
        X, Y = numpy.meshgrid(x, y)
        new_r = numpy.sqrt(X*X + Y*Y)
        new_t = numpy.arctan2(X, Y)

        ir = interp1d(r, numpy.arange(len(r)), bounds_error=False)
        it = interp1d(t, numpy.arange(len(t)))

        new_ir = ir(new_r.ravel())
        new_it = it(new_t.ravel())

        new_ir[new_r.ravel() > r.max()] = len(r)-1
        new_ir[new_r.ravel() < r.min()] = 0

        return numpy.array([new_ir, new_it])

    def polar2cartesian(self, grid, coordinates, shape):
        r = shape[0] - 1
        psf_cart = numpy.empty([2*r + 1, 2*r + 1])
        psf_cart[r:, r:] = map_coordinates(grid, coordinates, order=0).reshape(shape)
        psf_cart[r:,:r] = psf_cart[r:,:r:-1]
        psf_cart[:r,:] = psf_cart[:r:-1,:]
        return psf_cart

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

    def output_frames(self, rng, dataset, pathto='./images', image_fmt='image_%07d.npy', true_fmt='true_%07d.npy'):
        # Check and create the folders for image and output files.
        if not os.path.exists(pathto):
            os.makedirs(pathto)

        # spatiocyte_data, num_timesteps, index0, count_array_size = (
        #     dataset.data, dataset.index_array_size, dataset.index0, dataset.count_array_size)

        # set Fluorophores PSF
        self.set_fluo_psf(dataset)

        # num_timesteps = self.configs.shutter_index_array_size
        # index0 = self.configs.shutter_index_array_first

        if self.get_nprocs() == 1:
            self.output_frames_each_process(rng, dataset, pathto, image_fmt, true_fmt, dataset.index0, dataset.index_array_size)
        else:
            num_processes = multiprocessing.cpu_count()
            n, m = divmod(num_timesteps, num_processes)
            # when 10 tasks is distributed to 4 processes,
            # number of tasks of each process must be [3, 3, 2, 2]
            chunks = [n + 1 if i < m else n for i in range(num_processes)]

            processes = []
            start_index = index0

            for chunk in chunks:
                stop_index = start_index + chunk

                #XXX: Initialize rng for each process
                process = multiprocessing.Process(
                    target=self.output_frames_each_process,
                    args=(rng, dataset, pathto, image_fmt, true_fmt, start_index, stop_index))
                process.start()
                processes.append(process)
                start_index = stop_index

            for process in processes:
                process.join()

    def output_frames_each_process(self, rng, dataset, pathto, image_fmt, true_fmt, start_index, stop_index):
        # exposure time
        exposure_time = self.configs.detector_exposure_time

        # set delta_count
        delta_count = int(round(exposure_time / dataset.interval))

        for index in range(start_index, stop_index, 1):
            # frame-time in sec
            time = exposure_time * index
            _log.info('time: {} sec ({})'.format(time, index))

            c0 = (index - dataset.index0) * delta_count
            c1 = (index - dataset.index0 + 1) * delta_count

            camera, true_data = self.output_frame(rng, dataset, c0, c1, time)

            # save image to numpy-binary file
            image_file_name = os.path.join(pathto, image_fmt % (index))
            numpy.save(image_file_name, camera)

            # save true-dataset to numpy-binary file
            true_file_name = os.path.join(pathto, true_fmt % (index))
            numpy.save(true_file_name, true_data)

    def output_frame(self, rng, dataset, c0, c1, time):
        spatiocyte_data = dataset.data

        # focal point
        p_0 = self.get_focal_center(dataset.voxel_radius, dataset.lengths)

        # beam position: Assuming beam position = focal point (for temporary)
        p_b = copy.copy(p_0)

        # cell size (nm scale)
        _, Ny, Nz = self.get_cell_size(dataset.voxel_radius, dataset.lengths)

        # define cell in nm-scale
        cell = numpy.zeros(shape=(Nz, Ny))

        # # set delta_count
        # exposure_time = self.configs.detector_exposure_time
        # delta_count = int(round(exposure_time / dataset.interval))
        # c0 = (index - dataset.index0) * delta_count
        # c1 = (index - dataset.index0 + 1) * delta_count
        frame_data = spatiocyte_data[c0: c1]

        # loop for frame data
        for i, (i_time, particles) in enumerate(frame_data):
            _log.info('     {:02d}-th frame: {} sec'.format(i, i_time))

            # define true-dataset in last-frame
            # [frame-time, m-ID, m-state, p-state, (depth,y0,z0), sqrt(<dr2>)]
            true_data = numpy.zeros(shape=(len(particles), 7))
            true_data[:, 0] = time

            # loop for particles
            for j, particle_j in enumerate(particles):
                self.get_molecule_plane(cell, j, particle_j, p_b, p_0, true_data, dataset)

        # convert image in pixel-scale
        camera, true_data = self.detector_output(rng, cell, true_data, dataset)
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

    def detector_output(self, rng, cell, true_data, dataset):
        # Detector Output
        voxel_radius = dataset.voxel_radius
        voxel_size = (2.0*voxel_radius)/1e-9

        Nw_pixel, Nh_pixel = self.configs.detector_image_size

        pixel_length = self.configs.detector_pixel_length / self.configs.image_magnification
        image_scaling = pixel_length / (2.0 * voxel_radius)
        Np = image_scaling * voxel_size
        # Np = self.configs.image_scaling*voxel_size

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
        w0 = Nw_dummy//2 - int(Nw_cell/Np*(0.5-f_z))
        h0 = Nh_dummy//2 - int(Nh_cell/Np*(0.5-f_y))

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

    def get_nprocs(self):
        return self.__nprocs

    def set_fluo_psf(self, dataset):
        spatiocyte_data = dataset.data

        depths = set()

        # get cell size
        Nx, Ny, Nz = self.get_cell_size(dataset.voxel_radius, dataset.lengths)

        # count_array_size = len(self.configs.shutter_count_array)

        exposure_time = self.configs.detector_exposure_time
        # data_interval = self.configs.spatiocyte_interval

        delta_count = int(round(exposure_time / dataset.interval))

        for count in range(0, len(spatiocyte_data), delta_count):

            # focal point
            f_0 = self.configs.detector_focal_point
            p_0 = numpy.array([Nx, Ny, Nz])*f_0
            x_0, y_0, z_0 = p_0

            frame_data = spatiocyte_data[count: count + delta_count]

            for _, data in frame_data:
                for data_j in data:

                    p_i = numpy.array(data_j[0])/1e-9

                    # Snell's law
                    #amplitude, penet_depth = self.snells_law(p_i, p_0)

                    # Particle coordinte in real(nm) scale
                    p_i, radial, depth = self.get_coordinate(p_i, p_0)

                    # fluorophore axial position
                    fluo_depth = depth if depth < len(self.configs.depth) else -1

                    depths.add(int(fluo_depth))

        depths = list(depths)

        if (len(depths) > 0):
            if self.get_nprocs() != 1:
                self.fluo_psf = {}
                num_processes = min(multiprocessing.cpu_count(), len(depths))

                n, m = divmod(len(depths), num_processes)

                chunks = [n + 1 if i < m else n for i in range(num_processes)]

                processes = []
                start_index = 0

                for chunk in chunks:
                    stop_index = start_index + chunk
                    p, c = multiprocessing.Pipe()
                    process = multiprocessing.Process(
                        target=self.get_fluo_psf_process,
                        args=(depths[start_index:stop_index], c))
                    processes.append((p, process))
                    start_index = stop_index

                for _, process in processes:
                    process.start()

                for pipe, process in processes:
                    self.fluo_psf.update(pipe.recv())
                    process.join()

            else:
                self.fluo_psf = self.get_fluo_psf(depths)

    def get_fluo_psf(self, depths):
        r = self.configs.radial

        theta = numpy.linspace(0, 90, 91)

        z = numpy.linspace(0, +r[-1], len(r))
        y = numpy.linspace(0, +r[-1], len(r))

        coordinates = self.polar2cartesian_coordinates(r, theta, z, y)
        psf_t = numpy.ones_like(theta)
        result = {}

        for depth in depths:
            psf_r = self.configs.fluorophore_psf[depth]
            psf_polar = numpy.array(list(map(lambda x: psf_t * x, psf_r)))
            result[depth] = self.polar2cartesian(psf_polar, coordinates, (len(r), len(r)))

        return result

    def get_fluo_psf_process(self, depths, pipe):
        pipe.send(self.get_fluo_psf(depths))

    def use_multiprocess(self):
        """Deprecated."""
        warnings.warn('This is no longer supported.')
        return (self.get_nprocs() != 1)
        # envname = 'ECELL_MICROSCOPE_SINGLE_PROCESS'
        # return not (envname in os.environ and os.environ[envname])
