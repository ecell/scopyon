
import sys
import os
import copy
import tempfile
import time
import math
import operator
import random
#import h5py
import ctypes
import multiprocessing

import scipy
import numpy

import parameter_configs
from effects_handler import PhysicalEffects
from epifm_handler import VisualizerError, EPIFMConfigs, EPIFMVisualizer

from scipy.special import j0, gamma
from scipy.misc    import toimage


class FCSConfigs(EPIFMConfigs) :

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
        configs_dict = parameter_configs.__dict__.copy()
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



    def set_Pinhole(self, radius = None) :

        print('--- Pinhole :')

        self._set_data('pinhole_radius', radius)

        print('\tRadius = ', self.pinhole_radius, 'm')



    def set_Detector(self, detector = None,
                   mode = None,
                   pixel_length = None,
                   focal_point = None,
                   base_position = None,
                   exposure_time = None,
                   QE = None,
                   readout_noise = None,
                   dark_count = None,
                   gain = None,
                   dyn_stages = None,
                   pair_pulses = None
                   ):

        self._set_data('detector_switch', True)
        self._set_data('detector_type', detector)
        self._set_data('detector_mode', mode)
        self._set_data('detector_pixel_length', pixel_length)
        self._set_data('detector_focal_point', focal_point)
        self._set_data('detector_base_position', base_position)
        self._set_data('detector_exposure_time', exposure_time)
        self._set_data('detector_qeff', QE)
        self._set_data('detector_readout_noise', readout_noise)
        self._set_data('detector_dark_count', dark_count)
        self._set_data('detector_gain', gain)
        self._set_data('detector_dyn_stages', dyn_stages)
        self._set_data('detector_pair_pulses', pair_pulses)

        print('--- Detector : ', self.detector_type, ' (', self.detector_mode, 'mode )')
        print('\tPixel Size  = ', self.detector_pixel_length, 'm/pixel')
        print('\tFocal Point = ', self.detector_focal_point)
        print('\tPosition    = ', self.detector_base_position)
        print('\tExposure Time = ', self.detector_exposure_time, 'sec/image')
        print('\tQuantum Efficiency = ', 100*self.detector_qeff, '%')
        print('\tReadout Noise = ', self.detector_readout_noise, 'electron')
        print('\tDark Count = ', self.detector_dark_count, 'electron/sec')
        print('\tGain = ', 'x', self.detector_gain)
        print('\tDynode = ', self.detector_dyn_stages, 'stages')
        print('\tPair-pulses = ', self.detector_pair_pulses, 'sec')



    def set_Illumination_path(self) :

        r = self.radial
        d = numpy.linspace(0, 20000, 20001)
        #d = self.depth

        # (plank const) * (speed of light) [joules meter]
        hc = 2.00e-25

        # Illumination
        w_0 = self.source_radius

        # flux [joules/sec=watt]
        P_0 = self.source_flux

        # single photon energy [joules]
        wave_length = self.source_wavelength*1e-9
        E_wl = hc/wave_length

        # photon flux [photons/sec]
        N_0 = P_0/E_wl

        # Beam width [m]
        w_z = w_0*numpy.sqrt(1 + ((wave_length*d*1e-9)/(numpy.pi*w_0**2))**2)

        # photon flux density [photon/(sec m^2)]
        self.source_flux_density = numpy.array(map(lambda x : 2*N_0/(numpy.pi*x**2)*numpy.exp(-2*(r*1e-9/x)**2), w_z))

        print('Photon Flux Density (Max) :', numpy.amax(self.source_flux_density))



    def set_Detection_path(self) :

        wave_length = self.psf_wavelength*1e-9

        # Magnification
        Mag = self.image_magnification

        # set voxel radius
        voxel_radius = self.spatiocyte_VoxelRadius

        # set pixel length
        #pixel_length = (2.0*self.pinhole_radius)/Mag
        pixel_length = self.detector_pixel_length

        # set image scaling factor
        self.image_resolution = pixel_length
        self.image_scaling = pixel_length/(2.0*voxel_radius)

        print('Magnification : x %d' % (Mag))
        print('Resolution :', self.image_resolution, 'm/pixel')
        print('Scaling :', self.image_scaling)

        # Detector PSF
        self.set_PSF_detector()




class FCSVisualizer(EPIFMVisualizer) :

    '''
    Confocal Visualization class of e-cell simulator
    '''

    def __init__(self, configs=FCSConfigs(), effects=PhysicalEffects()) :

        assert isinstance(configs, FCSConfigs)
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



    def get_signal(self, time, pid, s_index, p_i, p_b, p_0, norm) :

        # set focal point
        x_0, y_0, z_0 = p_0

        # set source center
        x_b, y_b, z_b = p_b

        # set particle position
        x_i, y_i, z_i = p_i

        #
        r = self.configs.radial
        d = self.configs.depth

        # beam axial position
        d_s = abs(x_i - x_b)

        if (d_s < 20000) :
            source_depth = d_s
        else :
            source_depth = 19999

        # beam horizontal position (y-direction)
        hh = numpy.sqrt((y_i-y_b)**2)

        if (hh < len(r)) :
            source_horizon = hh
        else :
            source_horizon = r[-1]

        # get illumination PSF
        source_psf = self.configs.source_flux_density[int(source_depth)][int(source_horizon)]
        #source_max = norm*self.configs.source_flux_density[0][0]

        # signal conversion :
        #Intensity = self.get_intensity(time, pid, source_psf, source_max)
        Ratio = self.effects.conversion_ratio

        # fluorophore axial position
        d_f = abs(x_i - x_b)

        if (d_f < len(d)) :
            fluo_depth = d_f
        else :
            fluo_depth = d[-1]

        # get fluorophore PSF
        fluo_psf = self.fluo_psf[int(fluo_depth)]

        # signal conversion : Output PSF = PSF(source) * Ratio * PSF(Fluorophore)
        signal = norm * source_psf * Ratio * fluo_psf

        return signal



    def get_molecule_plane(self, cell, time, data, pid, p_b, p_0) :

        #voxel_size = (2.0*self.configs.spatiocyte_VoxelRadius)/1e-9

        # get beam position
        x_b, y_b, z_b = p_b

        # cutoff randius
        Mag = self.configs.image_magnification
        pinhole_radius = int(self.configs.pinhole_radius/Mag/1e-9)
        cut_off = int(1.5*pinhole_radius)

        # particles coordinate, species and lattice IDs
        c_id, s_id, l_id = data

        sid_array = numpy.array(self.configs.spatiocyte_species_id)
        s_index = (numpy.abs(sid_array - int(s_id))).argmin()

        if self.configs.spatiocyte_observables[s_index] is True :

            # Normalization
            unit_time = 1.0
            unit_area = (1e-9)**2
            norm = (unit_area*unit_time)/(4.0*numpy.pi)

            # particles coordinate in nm-scale
            p_i = self.get_coordinate(c_id)
            #p_i = p_0
            x_i, y_i, z_i = p_i

            if (numpy.sqrt((y_i - y_b)**2 + (z_i - z_b)**2) < cut_off) :

                #print pid, s_id, p_i
                # get signal matrix
                signal = self.get_signal(time, pid, s_index, p_i, p_b, p_0, norm)

                # add signal matrix to image plane
                self.overwrite_signal(cell, signal, p_i, p_b)



    def overwrite_signal(self, cell, signal, p_i, p_b) :

        # particle position
        x_i, y_i, z_i = p_i

        # beam position
        x_b, y_b, z_b = p_b

        # z-axis
        Nz_cell   = len(cell)
        Nz_signal = len(signal)

        Nh_cell = Nz_cell/2
        Nh_signal = (Nz_signal-1)/2

        z_to   = (z_i + Nh_signal) - (z_b - Nh_cell)
        z_from = (z_i - Nh_signal) - (z_b - Nh_cell)

        flag_z = True

        if (z_to > Nz_cell + Nz_signal) :
            flag_z = False

        elif (z_to > Nz_cell and
              z_to < Nz_cell + Nz_signal) :

            dz_to = z_to - Nz_cell

            zi_to = int(Nz_signal - dz_to)
            zb_to = int(Nz_cell)

        elif (z_to > 0 and z_to < Nz_cell) :

            zi_to = int(Nz_signal)
            zb_to = int(z_to)

        else : flag_z = False

        if (z_from < 0) :

            zi_from = int(abs(z_from))
            zb_from = 0

        else :

            zi_from = 0
            zb_from = int(z_from)

        if (flag_z == True) :
            ddz = (zi_to - zi_from) - (zb_to - zb_from)

            if (ddz > 0) : zi_to = zi_to - ddz
            if (ddz < 0) : zb_to = zb_to + ddz

        # y-axis
        Ny_cell  = cell.size/Nz_cell
        Ny_signal = signal.size/Nz_signal

        Nh_cell = Ny_cell/2
        Nh_signal = (Ny_signal-1)/2

        y_to   = (y_i + Nh_signal) - (y_b - Nh_cell)
        y_from = (y_i - Nh_signal) - (y_b - Nh_cell)

        flag_y = True

        if (y_to > Ny_cell + Ny_signal) :
            flag_y = False

        elif (y_to > Ny_cell and
              y_to < Ny_cell + Ny_signal) :

            dy_to = y_to - Ny_cell

            yi_to = int(Ny_signal - dy_to)
            yb_to = int(Ny_cell)

        elif (y_to > 0 and y_to < Ny_cell) :

            yi_to = int(Ny_signal)
            yb_to = int(y_to)

        else : flag_y = False

        if (y_from < 0) :

            yi_from = int(abs(y_from))
            yb_from = 0

        else :

            yi_from = 0
            yb_from = int(y_from)

        if (flag_y == True) :
            ddy = (yi_to - yi_from) - (yb_to - yb_from)

            if (ddy > 0) : yi_to = yi_to - ddy
            if (ddy < 0) : yb_to = yb_to + ddy

        if (flag_z == True and flag_y == True) :
#                   if (abs(ddy) > 2 or abs(ddz) > 2) :
#                       print zi_from, zi_to, yi_from, yi_to, ddy, ddz
#                       print zb_from, zb_to, yb_from, yb_to

            # add to cellular plane
            cell[zb_from:zb_to, yb_from:yb_to] += signal[zi_from:zi_to, yi_from:yi_to]

        #return cell



    def output_frames(self, num_div=1):

            # set Fluorophores PSF
        self.set_fluo_psf()

        start = self.configs.spatiocyte_start_time
        end = self.configs.spatiocyte_end_time

        exposure_time = self.configs.detector_exposure_time
        num_timesteps = int(math.ceil((end - start) / exposure_time))

        index0  = int(round(start/exposure_time))

        envname = 'ECELL_MICROSCOPE_SINGLE_PROCESS'

        if envname in os.environ and os.environ[envname]:
            self.output_frames_each_process(index0, num_timesteps)
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
                process = multiprocessing.Process(
                    target=self.output_frames_each_process,
                    args=(start_index, stop_index))
                process.start()
                processes.append(process)
                start_index = stop_index

            for process in processes:
                process.join()



    def output_frames_each_process(self, start_count, stop_count):

        voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9

        # cells dimenssion in nm-scale
        Nz = int(self.configs.spatiocyte_lengths[2] * voxel_size)
        Ny = int(self.configs.spatiocyte_lengths[1] * voxel_size)
        Nx = int(self.configs.spatiocyte_lengths[0] * voxel_size)

        # pixel length : nm/pixel
        Np = int(self.configs.image_scaling*voxel_size)

        # focal point
        p_0 = numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point

        # Beam position :
        beam_center = numpy.array(self.configs.detector_focal_point)

#                # set boundary condition
#                if (self.configs.spatiocyte_bc_switch == True) :
#
#                    bc = numpy.zeros(shape=(Nz, Ny))
#                    bc = self.set_boundary_plane(bc, p_b, p_0)

        # exposure time
        exposure_time = self.configs.detector_exposure_time

        # spatiocyte start time
        start_time = self.configs.spatiocyte_start_time

        time = exposure_time * start_count
        end  = exposure_time * stop_count

        # data-time interval
        data_interval = self.configs.spatiocyte_interval

        delta_time = int(round(exposure_time / data_interval))

        # create frame data composed by frame element data
        count  = start_count
        count0 = int(round(start_time / exposure_time))

        # initialize Physical effects
        #length0 = len(self.configs.spatiocyte_data[0][1])
        #self.effects.set_states(t0, length0)

        while (time < end) :

            image_file_name = os.path.join(self.configs.image_file_dir,
                                self.configs.image_file_name_format % (count))

            print('time : ', time, ' sec (', count, ')')

            # define cell in nm-scale
            #cell = numpy.zeros(shape=(Nz, Ny))
            # define image array in pixel-scale
            image_pixel = numpy.zeros([2])

            count_start = (count - count0)*delta_time
            count_end   = (count - count0 + 1)*delta_time

            frame_data = self.configs.spatiocyte_data[count_start:count_end]

            if (len(frame_data) > 0) :

                # Beam position : Fixed
                p_b = numpy.array([Nx, Ny, Nz])*beam_center
                x_b, y_b, z_b = p_b

                # loop for frame data
                i_time, i_data = frame_data[0]

                diff = abs(i_time - time)
                data = i_data

                for i, (i_time, i_data) in enumerate(frame_data) :
                    #print '\t', '%02d-th frame : ' % (i), i_time, ' sec'
                    i_diff = abs(i_time - time)

                    if (i_diff < diff) :
                        diff = i_diff
                        data = i_data

                # overwrite the scanned region to cell
                #r_p = int(self.configs.image_scaling*voxel_size/2)
                Mag = self.configs.image_magnification
                r_p = int(self.configs.pinhole_radius/Mag/1e-9)

                if (y_b-r_p < 0) : y_from = int(y_b)
                else : y_from = int(y_b - r_p)

                if (y_b+r_p >= Ny) : y_to = int(y_b)
                else : y_to = int(y_b + r_p)

                if (z_b-r_p < 0) : z_from = int(z_b)
                else : z_from = int(z_b - r_p)

                if (z_b+r_p >= Nz) : z_to = int(z_b)
                else : z_to = int(z_b + r_p)

                mask = numpy.zeros(shape=(z_to-z_from, y_to-y_from))

                zz, yy = numpy.ogrid[z_from-int(z_b):z_to-int(z_b), y_from-int(y_b):y_to-int(y_b)]
                rr_cut = yy**2 + zz**2 < r_p**2
                mask[rr_cut] = 1

                scan_cell = numpy.zeros_like(mask)

                # loop for particles
                for j, j_data in enumerate(data) :
                    self.get_molecule_plane(scan_cell, i_time, j_data, j, p_b, p_0)

                Photon_flux = numpy.sum(mask*scan_cell)

                # output tuple for particles
                image = self.detector_output((time, Photon_flux))

                # save data to numpy-binary file
                image_file_name = os.path.join(self.configs.image_file_dir, \
                                self.configs.image_file_name_format % (count))
                numpy.save(image_file_name, image)


            time  += exposure_time
            count += 1



    def prob_analog(self, y, alpha) :

        # get average gain
        A  = self.configs.detector_gain

        # get dynodes stages
        nu = self.configs.detector_dyn_stages

        B = 0.5*(A - 1)/(A**(1.0/nu) - 1)
        c = numpy.exp(alpha*(numpy.exp(-A/B) - 1))

        m_y = alpha*A
        m_x = m_y/(1 - c)

        s2_y = alpha*(A**2 + 2*A*B)
        s2_x = s2_y/(1 - c) - c*m_x**2

        #if (y < 10*A) :
        # Rayleigh approximation
        #s2 = (2.0/numpy.pi)*m_x**2
        #prob = y/s2*numpy.exp(-0.5*y**2/s2)

        # probability distribution
        prob = numpy.zeros(shape=(len(y)))

        # get index
        k = (numpy.abs(10*A - y)).argmin()

        # Gamma approximation
        k_1 = m_x
        k_2 = (m_y**2 + s2_y)/(1 - c)

        a = 1/(k_1*(k_2/k_1**2 - 1))
        b = a*k_1

        prob[0:k] = a/gamma(b)*(a*y[0:k])**(b-1)*numpy.exp(-a*y[0:k])

        if (k < len(y)) :
            # Truncated Gaussian approximation
            Q = 0
            beta0 = m_x/numpy.sqrt(s2_x)
            beta  = beta0
            delta = 0.1*beta0

            while (beta < 11*beta0) :
                Q += numpy.exp(-0.5*beta**2)/numpy.sqrt(2*numpy.pi)*delta
                beta += delta

            prob[k:] = numpy.exp(-0.5*(y[k:] - m_x)**2/s2_x)/(numpy.sqrt(2*numpy.pi*s2_x)*(1 - Q))

        return prob




    def detector_output(self, array) :

        # Detector Output
        voxel_radius = self.configs.spatiocyte_VoxelRadius
        voxel_size = (2.0*voxel_radius)/1e-9

        # set seed for random number
        numpy.random.seed()

        # observational time
        T = self.configs.detector_exposure_time

        # conversion : photon --> photoelectron --> ADC count
        # pixel position
        pixel = (0, 0)

        # Detector : Quantum Efficiency
        #index = int(self.configs.psf_wavelength) - int(self.configs.wave_length[0])
        QE = self.configs.detector_qeff

        # get signal (photon flux and photons)
        time, Flux = array

        if (self.configs.detector_mode == "Photon-counting") :
            # pair-pulses time resolution (sec)
            t_pp = self.configs.detector_pair_pulses
            Flux = Flux/(1 + Flux*t_pp)

        Photons = Flux*T

        # get constant background
        if (self.effects.background_switch == True) :

            Photons_bg = self.effects.background_mean
            Photons += Photons_bg

        # get signal (expectation)
        Exp = QE*Photons

        # get dark count
        D = self.configs.detector_dark_count
        Exp += D*T

        # select Camera type
        if (self.configs.detector_mode == "Photon-counting") :

            # get signal (poisson distributions)
            signal = numpy.random.poisson(Exp, None)


        if (self.configs.detector_mode == "Analog") :

            # get signal (photoelectrons)
            if (Exp > 1e-8) :

                # get EM gain
                G = self.configs.detector_gain

                # signal array
                if (Exp > 1) : sig = numpy.sqrt(Exp)
                else : sig = 1

                s_min = int(G*(Exp - 10*sig))
                s_max = int(G*(Exp + 10*sig))

                if (s_min < 0) : s_min = 0

                delta = (s_max - s_min)/1000.

                s = numpy.array([k*delta+s_min for k in range(1000)])

                # probability density fuction
                #p_signal = numpy.array(map(lambda y : self.prob_analog(y, Exp), s))
                p_signal = self.prob_analog(s, Exp)
                p_ssum = p_signal.sum()

                # get signal (photoelectrons)
                signal = numpy.random.choice(s, None, p=p_signal/p_ssum)
                signal = signal/G

            else :
                signal = 0

        # get detector noise (photoelectrons)
        Nr = QE*self.configs.detector_readout_noise

        if (Nr > 0) : noise = numpy.random.normal(0, Nr, None)
        else : noise = 0

        # A/D converter : Photoelectrons --> ADC counts
        PE  = signal + noise
        ADC = self.get_ADC_value(pixel, PE)

        # set data in array
        output = [time, Exp, ADC]

        return output
