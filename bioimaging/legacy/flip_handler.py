
import sys
import os
import copy
import tempfile
import shutil
import time
import math
import operator
import random
import csv
#import h5py
import ctypes
import multiprocessing

import scipy
import numpy

import bleaching_configs
from effects_handler import PhysicalEffects
from lscm_handler import VisualizerError, LSCMConfigs, LSCMVisualizer

from scipy.special import j0, gamma
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
#from scipy.interpolate import RectBivariateSpline, BivariateSpline
from scipy.misc    import toimage


class FLIPConfigs(LSCMConfigs) :

    '''
    Point-scanning Confocal configuration

        Point-like Gaussian Profile
            +
        Point-scanning
            +
        Pinhole
            +
        Detector : PMT
    '''

    def __init__(self, user_configs_dict = None):

        # default setting
        configs_dict = bleaching_configs.__dict__.copy()
        #configs_dict = parameter_configs.__dict__.copy()
        #configs_dict_fcs = fcs_configs.__dict__.copy()
        #configs_dict.update(configs_dict_fcs)

        # user setting
        if user_configs_dict is not None:
            if type(user_configs_dict) != type({}):
                print('Illegal argument type for constructor of Configs class')
                sys.exit()
            configs_dict.update(user_configs_dict)

        for key, val in configs_dict.items():
            if key[0] != '_': # Data skip for private variables in setting_dict.
                if type(val) == type({}) or type(val) == type([]):
                    copy_val = copy.deepcopy(val)
                else:
                    copy_val = val
                setattr(self, key, copy_val)



    def set_LightSource(self,  source_type = None,
                                wave_length = None,
                                imaging_flux = None,
                                bleaching_flux = None,
                                bleaching_size = None,
                                bleaching_position = None,
                                bleaching_time = None,
                                radius = None,
                                angle  = None):

        self._set_data('source_switch', True)
        self._set_data('source_type', source_type)
        self._set_data('source_wavelength', wave_length)
        self._set_data('source_flux', imaging_flux)
        self._set_data('source_bleaching_flux', bleaching_flux)
        self._set_data('source_bleaching_size', bleaching_size)
        self._set_data('source_bleaching_position', bleaching_position)
        self._set_data('source_bleaching_time', bleaching_time)
        self._set_data('source_radius', radius)
        self._set_data('source_angle', angle)

        print('--- Light Source :', self.source_type)
        print('\tWave Length = ', self.source_wavelength, 'nm')
        print('\tImaging : Beam Flux = ', self.source_flux, 'W (=joule/sec)')
        print('\tBleaching : Beam Flux = ', self.source_bleaching_flux, 'W (=joule/sec)')
        print('\tBleaching : Size  = ', self.source_bleaching_size[0], 'x', self.source_bleaching_size[1], 'pixels')
        print('\tBleaching : Position = ', self.source_bleaching_position, 'pixel')
        print('\tBleaching : Time = ', self.source_bleaching_time, 'pixel')
        print('\t1/e2 Radius = ', self.source_radius, 'm')
        print('\tAngle = ', self.source_angle, 'degree')




    def set_Illumination_path(self) :

        r = numpy.linspace(0, 20000, 20001)
        d = numpy.linspace(0, 20000, 20001)

        # (plank const) * (speed of light) [joules meter]
        hc = 2.00e-25

        # Illumination
        w_0 = self.source_radius

        # flux for imaging [joules/sec=watt]
        P_0 = self.source_flux

        # flux for imaging [joules/sec=watt]
        P_1 = self.source_bleaching_flux

        # single photon energy [joules]
        wave_length = self.source_wavelength*1e-9
        E_wl = hc/wave_length

        # photon flux for imaging [photons/sec]
        N_0 = P_0/E_wl

        # photon flux for bleaching [photons/sec]
        N_1 = P_1/E_wl

        # Beam width [m]
        w_z = w_0*numpy.sqrt(1 + ((wave_length*d*1e-9)/(numpy.pi*w_0**2))**2)

        # photon flux density for imaging [photon/(sec m^2)]
        self.source_flux_density = numpy.array(map(lambda x : 2*N_0/(numpy.pi*x**2)*numpy.exp(-2*(r*1e-9/x)**2), w_z))

        # photon flux density for bleaching [photon/(sec m^2)]
        self.source_bleaching_flux_density = numpy.array(map(lambda x : 2*N_1/(numpy.pi*x**2)*numpy.exp(-2*(r*1e-9/x)**2), w_z))

        print('Photon Flux Density (Imaging) :', numpy.amax(self.source_flux_density))
        print('Photon Flux Density (Bleaching) :', numpy.amax(self.source_bleaching_flux_density))




class FLIPVisualizer(LSCMVisualizer) :

    '''
    Confocal Visualization class of e-cell simulator
    '''

    def __init__(self, configs=FLIPConfigs(), effects=PhysicalEffects()) :

        assert isinstance(configs, FLIPConfigs)
        self.configs = configs

        assert isinstance(effects, PhysicalEffects)
        self.effects = effects

        """
        Check and create the folder for image file.
        """
        if not os.path.exists(self.configs.image_file_dir):
            os.makedirs(self.configs.image_file_dir)
        #else:
        #    for file in os.listdir(self.configs.movie_image_file_dir):
        #       os.remove(os.path.join(self.configs.movie_image_file_dir, file))

        """
        Optical Path
        """
        self.configs.set_Optical_path()



    def rewrite_InputData(self, output_file_dir=None) :

        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)

        # define observational image plane in nm-scale
        voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9

        # image dimenssion in pixel-scale
        Nw_pixel = self.configs.detector_image_size[0]
        Nh_pixel = self.configs.detector_image_size[1]

        ## cell size (nm scale)
        Nz = int(self.configs.spatiocyte_lengths[2]*voxel_size)
        Ny = int(self.configs.spatiocyte_lengths[1]*voxel_size)
        Nx = int(self.configs.spatiocyte_lengths[0]*voxel_size)

        # pixel length : nm/pixel
        Np = int(self.configs.image_scaling*voxel_size)

        # image dimenssion in pixel-scale
        Nz_pixel = Nz/Np
        Ny_pixel = Ny/Np

        # Active states per detection time
        dt = self.configs.spatiocyte_interval
        exposure_time = self.configs.detector_exposure_time

        start = self.configs.spatiocyte_start_time
        end = self.configs.spatiocyte_end_time

        # Abogadoro's number
        Na = self.effects.avogadoros_number

        # spatiocyte size
        radius = self.configs.spatiocyte_VoxelRadius
        volume = 4.0/3.0*numpy.pi*radius**3
        depth  = 2.0*radius

        # Quantum yield
        QY = self.effects.quantum_yield

        # Abs coefficient [1/(cm M)]
        abs_coeff = self.effects.abs_coefficient

        # Imaging : photon flux density [#photons/(sec m2)]
        n0 = self.configs.source_flux_density

        # Bleaching : photon flux density [#photons/(sec m2)]
        n1 = self.configs.source_bleaching_flux_density

        # scan time per pixel area
        dT = exposure_time/(Nh_pixel*Nw_pixel)

        # Cross-Section [m2]
        xsec = numpy.log(10)*(0.1*abs_coeff/Na)

        # Imaging : the number of absorbed photons
        N_abs0 = xsec*n0*dT

        # Bleaching : the number of absorbed photons
        N_abs1 = xsec*n1*dT

        # Beer-Lambert law : A = log(I0/I) = coef * concentration * path-length
        A = (abs_coeff*0.1/Na)*(1/volume)*depth

        # Imaging : the number of emitted photons
        N_emit0 = QY*N_abs0*(1 - 10**(-A))

        # Bleaching : the number of emitted photons
        N_emit1 = QY*N_abs1*(1 - 10**(-A))

        # copy input file
        csv_input = self.configs.spatiocyte_file_directry + '/pt-input.csv'
        shutil.copyfile(csv_input, output_file_dir + '/pt-input.csv')

        # read input file
        csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (0)
        csv_list = list(csv.reader(open(csv_file_path, 'r')))

        # fluorescence
        if (self.effects.photobleaching_switch == True) :

            # get the number of particles
            N_particles = len(csv_list)

            # set fluorescence
            self.effects.set_photophysics_4lscm(start, end, dT, N_particles)

            bleach = self.effects.fluorescence_bleach
            budget = self.effects.fluorescence_budget*N_emit0[0][0]
            state  = self.effects.fluorescence_state

            scan_state = numpy.copy(state*0)
            scan_counter = numpy.zeros(shape=(N_particles))

        # Mesh-grid for pixel-image
        w = numpy.linspace(0, Nw_pixel-1, Nw_pixel)
        h = numpy.linspace(0, Nh_pixel-1, Nh_pixel)

        W, H = numpy.meshgrid(w, h)

        # scan-time (sec) per pixel-image
        dt_w = exposure_time/(Nh_pixel*Nw_pixel)
        dt_h = dt_w*Nw_pixel

        T_b = start + W*dt_w + H*dt_h
        time = T_b.reshape((Nw_pixel*Nh_pixel))

        # cell-center in pixel-image
        W0, H0 = Nw_pixel/2, Nh_pixel/2

        # cell-origin in pixel-image
        w0, h0 = W0-int(Ny_pixel/2), H0-int(Nz_pixel/2)

        # cell-located region in pixel image
        cell = numpy.zeros((Nw_pixel, Nh_pixel))
        cell[w0:w0+Ny_pixel,h0:h0+Nz_pixel] = 1
        cell_bool = cell.reshape((Nw_pixel*Nh_pixel))

        # Imaging : beam-position (nm) relative to cell-origin
        Y_b, Z_b = numpy.meshgrid(Np*(w-w0), Np*(h-h0))
        X_b = Nx*self.configs.detector_focal_point[0]

        # Bleaching : area size in pixel-image
        Nw_bleach = self.configs.source_bleaching_size[0]
        Nh_bleach = self.configs.source_bleaching_size[1]

        # Bleaching : beam position in pixel-image
        Wb, Hb = self.configs.source_bleaching_position
        w_b0, h_b0 = Wb-int(Nw_bleach/2), Hb-int(Nh_bleach/2)

        # Bleaching : region in pixel image
        bleach_area = numpy.zeros((Nw_pixel, Nh_pixel))
        bleach_area[w_b0:w_b0+Nw_bleach,h_b0:h_b0+Nh_bleach] = 1
        bleach_bool = bleach_area.reshape((Nw_pixel*Nh_pixel))

        # set flux density as a function of radial and depth
        r = numpy.linspace(0, 20000, 20001)
        d = numpy.linspace(0, 20000, 20001)

        # loop for scannig
        index0 = int(start/dt)
        delta_index = int(exposure_time/dt)

        # set bleaching index
        time_bleaching  = self.configs.source_bleaching_time
        count_bleaching = int(time_bleaching/exposure_time)

        while (time[-1] < end) :

            count_imaging = int(round(time[0]/exposure_time))

            print('time : ', time[0], '-', time[-1], ' sec/image (', count_imaging, ')')

            # set frame datasets
            index1 = int(round(time[0]/dt))
            index_start = (index1 - index0)
            index_end   = (index1 - index0) + delta_index

            time_index = (time/dt).astype('int')

            for index in range(index_start, index_end, 1) :

                # read input file
                csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (index)

                csv_list = list(csv.reader(open(csv_file_path, 'r')))
                dataset  = numpy.array(csv_list)

                # set imaging-area
                time_bool = (time_index == index).astype('int')
                imaging = (time_bool*cell_bool).astype('int')
                len_imaging = len(imaging[imaging > 0])

                print(index, index*dt+start, 'length :', len_imaging)

                # set photo-bleaching-area
                non_bleached_area = (time_bool*cell_bool*(1 - bleach_bool)).astype('int')
                bleached_area = (time_bool*cell_bool*bleach_bool).astype('int')

                if (len_imaging > 0) :

                    # loop for particles
                    for j, data_j in enumerate(dataset) :

                        # set particle position
                        c_id = (float(data_j[1]), float(data_j[2]), float(data_j[3]))
                        x_i, y_i, z_i = self.get_coordinate(c_id)

                        # beam axial position
                        """ Distance between beam depth and fluorophore (depth) """
                        dX = int(abs(X_b - x_i))
                        depth = dX if dX < d[-1] else d[-1]

                        # beam lateral position
                        """ Distance between beam position and fluorophore (plane) """
                        dR = numpy.sqrt((Y_b - y_i)**2 + (Z_b - z_i)**2)
                        dR_array = dR.reshape((Nw_pixel*Nh_pixel))

                        # Photobleaching
                        if (count_imaging >= count_bleaching) :
                            # set flux density as a function of radial and depth (at Non-bleached area)
                            R0 = (dR_array*non_bleached_area).astype('int')
                            radius0 = R0[R0 > 0]

                            func_emit0 = interp1d(r, N_emit0[depth], bounds_error=False, fill_value=N_emit0[depth][r[-1]])
                            N_emit = func_emit0(radius0)

                            # set flux density as a function of radial and depth (at Bleached area)
                            R1 = (dR_array*bleached_area).astype('int')
                            radius1 = R1[R1 > 0]

                            func_emit1 = interp1d(r, N_emit1[depth], bounds_error=False, fill_value=N_emit1[depth][r[-1]])
                            N_emit_bleaching = func_emit1(radius1)

                            # the number of emitted photons
                            N_emit_sum = N_emit.sum() + N_emit_bleaching.sum()

                        # Imaging
                        else :
                            # set flux density as a function of radial and depth (at Non-bleached area)
                            R0 = (dR_array*imaging).astype('int')
                            radius0 = R0[R0 > 0]

                            func_emit0 = interp1d(r, N_emit0[depth], bounds_error=False, fill_value=N_emit0[depth][r[-1]])
                            N_emit = func_emit0(radius0)

                            # the number of emitted photons
                            N_emit_sum = N_emit.sum()

                        scan_counter[j] += (N_emit_sum/N_emit0[0][0])

                        if (int(scan_counter[j]) < len(state[j,:])) :
                            state_j = state[j,int(scan_counter[j])]
                        else :
                            state_j = state[j,-1]

                        if (budget[j] > N_emit_sum) :
                            budget[j] = budget[j] - state_j*N_emit_sum
                        else :
                            budget[j] = 0

                        scan_state[j,count_imaging-int(start/dt)] = state_j


                state_stack = numpy.column_stack((scan_state[:,count_imaging-int(start/dt)], (budget/N_emit0[0][0]).astype('int')))
                new_dataset = numpy.column_stack((dataset, state_stack))

                # write output file
                output_file = output_file_dir + '/pt-%09d.0.csv' % (index)

                with open(output_file, 'w') as f :
                    writer = csv.writer(f)
                    writer.writerows(new_dataset)

            time  += exposure_time
