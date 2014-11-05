
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
import parameter_effects
from effects_handler import PhysicalEffects

from epifm_handler import VisualizerError, EPIFMConfigs, EPIFMVisualizer

from time import sleep
from scipy.special import j0
from scipy.misc    import toimage


class FCSConfigs(EPIFMConfigs) :

    '''
    FCS configuration

	Point-like Gaussian Beam
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
                print 'Illegal argument type for constructor of Configs class'
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

        print '--- Pinhole :'

        self._set_data('pinhole_radius', radius)

        print '\tPinhole Radius = ', self.pinhole_radius, 'm'



    def set_Illumination_path(self) :

        #r = self.radial
        #d = self.depth
	r = numpy.linspace(0, 20000, 20001)
	d = numpy.linspace(0, 20000, 20001)

        # (plank const) * (speed of light) [joules meter]
        hc = 2.00e-25

	# Illumination
        w_0 = self.source_radius

	# power [joules/sec]
        P_0 = self.source_power

	# illumination area [m^2]
	A_0 = numpy.pi*w_0**2

        # single photon energy
        wave_length = self.source_wavelength*1e-9
        E_wl = hc/wave_length

	# photon flux [photons/sec]
        N_0 = P_0/E_wl

	# Rayleigh range
        z_R = numpy.pi*w_0**2/wave_length

        # Beam Flux [photons/(m^2 sec)]
        w_z = w_0*numpy.sqrt(1 + ((wave_length*d*1e-9)/(numpy.pi*w_0**2))**2)

	# photon flux density [photon/(sec m^2)]
        self.source_flux = numpy.array(map(lambda x : 2*N_0/(numpy.pi*x**2)*numpy.exp(-2*(r*1e-9/x)**2), w_z))

	print 'Photon FLux Density (Max) :', numpy.amax(self.source_flux)



    def set_Detection_path(self) :

        wave_length = self.psf_wavelength*1e-9

	# Magnification
	Mag = self.image_magnification

	# set image scaling factor
        voxel_radius = self.spatiocyte_VoxelRadius

	# set pinhole pixel length
	pixel_length = (2.0*self.pinhole_radius)/Mag

	self.image_resolution = pixel_length
	self.image_scaling = pixel_length/(2.0*voxel_radius)

	print 'Resolution :', self.image_resolution, 'm'
	print 'Scaling :', self.image_scaling

        # Detector PSF
        self.set_PSF_detector()




class FCSVisualizer(EPIFMVisualizer) :

	'''
	FCS Visualization class of e-cell simulator
	'''

	def __init__(self, configs=EPIFMConfigs(), effects=PhysicalEffects()) :

		assert isinstance(configs, EPIFMConfigs)
		self.configs = configs

                assert isinstance(effects, PhysicalEffects)
                self.effects = effects

		"""
		Check and create the folder for image file.
		"""
		if not os.path.exists(self.configs.image_file_dir):
		    os.makedirs(self.configs.image_file_dir)
		#else:
		#    for file in os.listdir(self.configs.image_file_dir):
		#	os.remove(os.path.join(self.configs.image_file_dir, file))

                """
                Optical Path
                """
		self.configs.set_Optical_path()



	def get_noise_analog(self, current) :

		# detector noise in current unit (Ampere)
                NA = self.configs.detector_readout
                Id = self.configs.detector_dark_current
                F  = self.configs.detector_excess
                B  = self.configs.detector_bandwidth
                M  = self.configs.detector_emgain
		e  = self.configs.electron_charge

                # Ref from Hamamatsu PMT technical guide
                sigma2 = 2*e*B*F*(M**2)*(current + 2*Id/M) + (NA)**2
                noise  = numpy.sqrt(sigma2)
        
                return noise



	def get_noise_pulse(self, signal_rate) :

		# detector noise in count rate unit (#/sec)
                Nr = self.configs.detector_readout
                Id = self.configs.detector_dark_current
                M  = self.configs.detector_emgain
		e  = self.configs.electron_charge

		# dark count rate (cathode)
		D  = Id/M/e

		# observational time
                B  = self.configs.detector_bandwidth
		T = 1/(2*B)

                # Ref from Hamamatsu PMT technical guide
                sigma2 = (signal_rate + 2*D)/T + (Nr)**2
                noise  = numpy.sqrt(sigma2)
        
                return noise



	def get_molecule_plane(self, cell, time, data, pid, p_b, p_0) :

		voxel_size = (2.0*self.configs.spatiocyte_VoxelRadius)/1e-9

		# get beam position
		x_b, y_b, z_b = p_b

                # cutoff randius
		PH_radius = int(self.configs.image_scaling*voxel_size/2)
		cut_off = 5*PH_radius

		# particles coordinate, species and lattice IDs
                c_id, s_id, l_id = data

		sid_array = numpy.array(self.configs.spatiocyte_species_id)
		s_index = (numpy.abs(sid_array - int(s_id))).argmin()

		if self.configs.spatiocyte_observables[s_index] is True :

		    # Normalization
		    unit_area = (1e-9)**2
		    norm = unit_area/(4.0*numpy.pi)

		    # particles coordinate in real(nm) scale
                    #pos = self.get_coordinate(c_id)
                    #p_i = numpy.array(pos)*voxel_size
                    p_i = self.get_coordinate(c_id)

                    #p_i = p_b
		    #D = 100e-12
		    #p_i[2] = p_b[2] + (numpy.sqrt(6*D*time) - numpy.sqrt(6*D*0.04))/1e-9

		    # get particle position
		    x_i, y_i, z_i = p_i

		    if ((y_i - y_b)**2 + (z_i - z_b)**2 < cut_off**2) :

        	    	# get signal matrix
        	    	signal = self.get_signal(time, pid, s_index, p_i, p_b, p_0, norm)

			#print pid, p_i, numpy.amax(signal)
			# add signal matrix to image plane
        	    	self.overwrite_signal(cell, signal, p_i)



	def output_frames(self, num_div=1):

	    # set Fluorophores PSF
	    self.set_fluo_psf()

            start = self.configs.spatiocyte_start_time
            end = self.configs.spatiocyte_end_time

	    bandwidth = self.configs.detector_bandwidth
            exposure_time = 1/(2*bandwidth)
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

                # define observational image plane in nm-scale
                voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9

                Nz = int(self.configs.spatiocyte_lengths[2] * voxel_size)
                Ny = int(self.configs.spatiocyte_lengths[1] * voxel_size)
                Nx = int(self.configs.spatiocyte_lengths[0] * voxel_size)

                # focal point
                p_0 = numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point

                # beam position : focal point
                p_b = copy.copy(p_0)
		x_b, y_b, z_b = p_b

                # set boundary condition
                if (self.configs.spatiocyte_bc_switch == True) :

                    bc = numpy.zeros(shape=(Nz, Ny))
                    bc = self.set_boundary_plane(bc, p_b, p_0)

                # exposure time
		bandwidth = self.configs.detector_bandwidth
                exposure_time  = 1/(2*bandwidth)

                spatiocyte_start_time = self.configs.spatiocyte_start_time
                time = exposure_time * start_count
                end  = exposure_time * stop_count

                # data-time interval
                data_interval = self.configs.spatiocyte_interval

                delta_time = int(round(exposure_time/data_interval))

                # create frame data composed by frame element data
                count  = start_count
                count0 = int(round(spatiocyte_start_time / exposure_time))

                # initialize Physical effects
                #length0 = len(self.configs.spatiocyte_data[0][1])
                #self.effects.set_states(t0, length0)

                while (time < end) :

                        # set image file name
                        image_file_name = os.path.join(self.configs.image_file_dir,
                                        self.configs.image_file_name_format % (count))

                        print 'time : ', time, ' sec (', count, ')'

                        # define cell
                        cell = numpy.zeros(shape=(Nz, Ny))

                        count_start = (count - count0)*delta_time
                        count_end   = (count - count0 + 1)*delta_time

                        frame_data = self.configs.spatiocyte_data[count_start:count_end]

                        # loop for frame data
                        for i, (i_time, data) in enumerate(frame_data):
                                print '\t', '%02d-th frame : ' % (i), i_time, ' sec'
                                # loop for particles
                                for j, data_j in enumerate(data):
                                        self.get_molecule_plane(cell, i_time, data_j, j, p_b, p_0)

			# Photon detection through pinhole
			r_p = int(self.configs.image_scaling*voxel_size/2)

			if (y_b-r_p < 0) : y_from = y_b
			else : y_from = y_b - r_p

			if (y_b+r_p > Ny) : y_to = y_b
			else : y_to = y_b + r_p

			if (z_b-r_p < 0) : z_from = z_b
			else : z_from = z_b - r_p

			if (z_b+r_p > Nz) : z_to = z_b
			else : z_to = z_b + r_p

			mask = numpy.zeros(shape=(z_to-z_from, y_to-y_from))

			zz, yy = numpy.ogrid[z_from-z_b:z_to-z_b, y_from-y_b:y_to-y_b]
			rr_cut = yy**2 + zz**2 < r_p**2
			mask[rr_cut] = 1

			# get photon flux (Photons/sec)
			photon_flux = numpy.sum(mask*cell[z_from:z_to, y_from:y_to])

                        # get outputs from analog and photon-counting modes
                        current, charge, pulse_rate, pulse, ADC = self.detector_output(photon_flux)

                        # write a file
                        output_file= self.configs.image_file_dir + '/output_%09d.dat' % (count)

                        line  = str(time)	+ '\t'
                        line += str(photon_flux) + '\t'
                        line += str(current)	+ '\t'
                        line += str(charge)	+ '\t'
                        line += str(pulse_rate)	+ '\t'
                        line += str(pulse)	+ '\t'
                        line += str(ADC)	+ '\n'

                        with open(output_file, 'w') as output :
                            output.write(line)

                        time  += exposure_time
                        count += 1




	def detector_output(self, photon_flux) :

		# reset random seed
		numpy.random.seed()

		# get Quantum Efficiency
		#index = int(self.configs.psf_wavelength) - int(self.configs.wave_length[0])
		#QE = self.configs.detector_qeff[index]
		QE = 0.3

		# get background (photoelectrons)
		background = 0

		# detector noise in current unit (Ampere)
                NA = self.configs.detector_readout
                Id = self.configs.detector_dark_current
                F  = self.configs.detector_excess
                B  = self.configs.detector_bandwidth
                G  = self.configs.detector_emgain
		e  = self.configs.electron_charge
		D  = Id/e

		# observational time
		T = 1/(2*B)

		if (self.configs.detector_mode == 'Pulse') :
		    # expectation
		    E = round(QE*photon_flux*T)

		    # get photoelectron-signal
		    signal = numpy.random.poisson(E, None)

		    # get dark count
		    dark = 0
		    #dark = numpy.random.exponential(G*(D*T), None)

		    # get readout noise
		    if (NA > 0) :
			noise = numpy.random.normal(0, NA, None)
		    else :
			noise = 0

		    # get total pulses
		    pulse = (signal + background) + dark + noise

		    # A/D converter : Pulse --> ADC counts
		    pixel = (0, 0)
		    ADC = self.get_ADC_value(pixel, pulse)

		return anode_current, charge, pulse_rate, pulse, ADC



#	def get_ADC_value(self, photoelectron) :
#
#	    # check non-linearity
#	    fullwell = self.configs.ADConverter_fullwell
#
#	    if (photoelectron > fullwell) :
#		photoelectron = fullwell
#
#            # convert photoelectron to ADC counts (Grayscale)
#            k = self.configs.ADConverter_gain[0][0]
#            ADC0 = self.configs.ADConverter_offset[0][0]
#            ADC_max = 2**self.configs.ADConverter_bit - 1
#
#            ADC = photoelectron/k + ADC0
#
#	    if (ADC > ADC_max) :
#		ADC = ADC_max
#
#	    if (ADC < 0) :
#		ADC = 0
#
#	    return int(ADC)


