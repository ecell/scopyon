
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

import parameter_configs
from effects_handler import PhysicalEffects
from fcs_handler import VisualizerError, FCSConfigs, FCSVisualizer

from scipy.special import j0, gamma
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
#from scipy.interpolate import RectBivariateSpline, BivariateSpline
from scipy.misc    import toimage


class LSCMConfigs(FCSConfigs) :

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



    def set_LightSource(self,  source_type = None,
				wave_length = None,
                                flux = None,
                                center = None,
                                radius = None,
                                angle  = None ) :

        self._set_data('source_switch', True)
        self._set_data('source_type', source_type)
        self._set_data('source_wavelength', wave_length)
        self._set_data('source_flux', flux)
        self._set_data('source_radius', radius)
        self._set_data('source_angle', angle)

        print '--- Light Source :', self.source_type
        print '\tWave Length = ', self.source_wavelength, 'nm'
        print '\tBeam Flux = ', self.source_flux, 'W'
        print '\t1/e2 Radius = ', self.source_radius, 'm'
        print '\tAngle = ', self.source_angle, 'degree'




    def set_Pinhole(self, radius = None) :

        print '--- Pinhole :'

        self._set_data('pinhole_radius', radius)

        print '\tRadius = ', self.pinhole_radius, 'm'



    def set_Detector(self, detector = None,
		   mode = None,
		   image_size = None,
                   pixel_length = None,
                   focal_point = None,
                   base_position = None,
                   scan_time = None,
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
        self._set_data('detector_image_size', image_size)
        self._set_data('detector_pixel_length', pixel_length)
        self._set_data('detector_focal_point', focal_point)
        self._set_data('detector_base_position', base_position)
        self._set_data('detector_exposure_time', scan_time)
        self._set_data('detector_qeff', QE)
        self._set_data('detector_readout_noise', readout_noise)
        self._set_data('detector_dark_count', dark_count)
        self._set_data('detector_gain', gain)
        self._set_data('detector_dyn_stages', dyn_stages)
        self._set_data('detector_pair_pulses', pair_pulses)

	print '--- Detector : ', self.detector_type, ' (', self.detector_mode, 'mode )'
        print '\tImage Size  = ', self.detector_image_size[0], 'x', self.detector_image_size[1]
        print '\tPixel Size  = ', self.detector_pixel_length, 'm/pixel'
        print '\tFocal Point = ', self.detector_focal_point
        print '\tPosition    = ', self.detector_base_position
        print '\tScan Time = ', self.detector_exposure_time, 'sec/image'
        print '\tQuantum Efficiency = ', 100*self.detector_qeff, '%'
	print '\tReadout Noise = ', self.detector_readout_noise, 'electron'
        print '\tDark Count = ', self.detector_dark_count, 'electron/sec'
        print '\tGain = ', 'x', self.detector_gain
        print '\tDynode = ', self.detector_dyn_stages, 'stages'
        print '\tPair-pulses = ', self.detector_pair_pulses, 'sec'



    def set_Illumination_path(self) :

        r = numpy.linspace(0, 20000, 20001)
        d = numpy.linspace(0, 20000, 20001)

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

	print 'Photon Flux Density (Max) :', numpy.amax(self.source_flux_density)



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
	self.image_scaling = pixel_length/(2.0*voxel_radius)

	print 'Magnification : x %d' % (Mag)
	print 'Resolution :', pixel_length, 'm/pixel'
	print 'Scaling :', self.image_scaling

        # Detector PSF
        self.set_PSF_detector()



    def set_Optical_path(self) :

	# (0) Data : Cell Model Sample 
	self.set_Time_arrays()
	self.set_Spatiocyte_data_arrays()

	# (1) Illumination path : Light source --> Cell Model Sample
	self.set_Illumination_path()

	# (2) Detection path : Cell Model Sample --> Detector
	self.set_Detection_path()



    def reset_InputData(self, csv_file_directry, start=0, end=None, observable=None) :

	print '--- Input Spatiocyte Data : ', csv_file_directry

	### header	
	f = open(csv_file_directry + '/pt-input.csv', 'r')
        header = f.readline().rstrip().split(',')
        header[:5] = [float(_) for _ in header[:5]]
        f.close()

        interval, lengths, voxel_r, species_info = header[0], (header[3:0:-1]), header[4], header[5:]

	species_id = range(len(species_info))
	species_index  = [_.split(':')[1].split(']')[0] for _ in species_info]
	species_radius = [float(_.split('=')[1]) for _ in species_info]

#	#####
#	count_start = int(round(start/interval))
#	count_end   = int(round(end/interval))
#
#	data = []
#
#	# read lattice file
#	for count in range(count_start, count_end, 1) :
#
#	    csv_file_path = csv_file_directry + '/pt-%09d.0.csv' % (count)
#            
#            try :
#
#                csv_file = open(csv_file_path, 'r')
#
#		dataset = []
#
#		for row in csv.reader(csv_file) :
#		    dataset.append(row)
#
#                ### particle data
#		time = float(dataset[0][0])
#
#		particles = []
#
#		from ast import literal_eval
#
#		for data_id in dataset :
#		    c_id = (float(data_id[1]), float(data_id[2]), float(data_id[3]))
#		    s_id, l_id = literal_eval(data_id[6])
#		    particles.append((c_id, s_id, l_id))
#                    
#		data.append([time, particles])
#
#
#            except Exception :
#                print 'Error : ', csv_file_path, ' not found'
#		exit()
#
#	data.sort(lambda x, y:cmp(x[0], y[0]))

        # get filename
        self._set_data('spatiocyte_file_directry', csv_file_directry)

        # get run time
        self._set_data('spatiocyte_start_time', start)
        self._set_data('spatiocyte_end_time', end)
        self._set_data('spatiocyte_interval', interval)

	# get data
        #self._set_data('spatiocyte_data', data)

        # get species properties
        self._set_data('spatiocyte_species_id', species_id)
        self._set_data('spatiocyte_index',  species_index)
	#self._set_data('spatiocyte_diffusion', species_diffusion)
	self._set_data('spatiocyte_radius', species_radius)

        # get lattice properties
        #self._set_data('spatiocyte_lattice_id', map(lambda x : x[0], lattice))
        self._set_data('spatiocyte_lengths', lengths)
        self._set_data('spatiocyte_VoxelRadius', voxel_r)
        self._set_data('spatiocyte_theNormalizedVoxelRadius', 0.5)

       
        # set observable
        if observable is None :
            index = [True for i in range(len(self.spatiocyte_index))]
        else :          
            index = map(lambda x :  True if x.find(observable) > -1 else False, self.spatiocyte_index)

	#index = [False, True]
        self.spatiocyte_observables = copy.copy(index)


        print '\tStart time =', self.spatiocyte_start_time, 'sec'
        print '\tEnd   time =', self.spatiocyte_end_time, 'sec'
        print '\tInterval   =', self.spatiocyte_interval, 'sec'
	print '\tVoxel radius =', self.spatiocyte_VoxelRadius, 'm'
        print '\tCompartment lengths :', self.spatiocyte_lengths, 'voxels'
        print '\tSpecies Index :', self.spatiocyte_index
        print '\tObservable :', self.spatiocyte_observables


	# Visualization error	
	if self.spatiocyte_species_id is None:
	    raise VisualizerError('Cannot find species_id in any given csv files')

#	if len(self.spatiocyte_data) == 0:
#	    raise VisualizerError('Cannot find spatiocyte_data in any given csv files: ' \
#	                    + ', '.join(csv_file_directry))

	if len(self.spatiocyte_index) == 0 :
	    raise VisualizerError('Cannot find spatiocyte_index in any given csv files: ' \
	                    + ', '.join(csv_file_directry))




    def set_InputData(self, csv_file_directry, start=0, end=None, observable=None) :

	print '--- Input Spatiocyte Data : ', csv_file_directry

	### header	
	f = open(csv_file_directry + '/pt-input.csv', 'r')
        header = f.readline().rstrip().split(',')
        header[:5] = [float(_) for _ in header[:5]]
        f.close()

        interval, lengths, voxel_r, species_info = header[0], (header[3:0:-1]), header[4], header[5:]

	species_id = range(len(species_info))
	species_index  = [_.split(':')[1].split(']')[0] for _ in species_info]
	species_radius = [float(_.split('=')[1]) for _ in species_info]

	#####
	count_start = int(round(start/interval))
	count_end   = int(round(end/interval))

	data = []

	# read lattice file
	for count in range(count_start, count_end, 1) :

	    csv_file_path = csv_file_directry + '/pt-%09d.0.csv' % (count)
            
            try :

                csv_file = open(csv_file_path, 'r')

		dataset = []

		for row in csv.reader(csv_file) :
		    dataset.append(row)

                ### particle data
		time = float(dataset[0][0])

		particles = []

		from ast import literal_eval

		for data_id in dataset :
		    c_id = (float(data_id[1]), float(data_id[2]), float(data_id[3]))
		    s_id, l_id = literal_eval(data_id[6])

		    try :
		        p_state, cyc_id = float(data_id[7]), float(data_id[8])
		    except Exception :
		        p_state, cyc_id = 1.0, float('inf')

		    particles.append((c_id, s_id, l_id, p_state, cyc_id))

		data.append([time, particles])


            except Exception :
                print 'Error : ', csv_file_path, ' not found'
		exit()

	data.sort(lambda x, y:cmp(x[0], y[0]))

        # get run time
        self._set_data('spatiocyte_start_time', start)
        self._set_data('spatiocyte_end_time', end)
        self._set_data('spatiocyte_interval', interval)

	# get data
        self._set_data('spatiocyte_data', data)

        # get species properties
        self._set_data('spatiocyte_species_id', species_id)
        self._set_data('spatiocyte_index',  species_index)
	#self._set_data('spatiocyte_diffusion', species_diffusion)
	self._set_data('spatiocyte_radius', species_radius)

        # get lattice properties
        #self._set_data('spatiocyte_lattice_id', map(lambda x : x[0], lattice))
        self._set_data('spatiocyte_lengths', lengths)
        self._set_data('spatiocyte_VoxelRadius', voxel_r)
        self._set_data('spatiocyte_theNormalizedVoxelRadius', 0.5)
       
        # set observable
        if observable is None :
            index = [True for i in range(len(self.spatiocyte_index))]
        else :          
            index = map(lambda x :  True if x.find(observable) > -1 else False, self.spatiocyte_index)

	#index = [False, True]
        self.spatiocyte_observables = copy.copy(index)

        print '\tStart time =', self.spatiocyte_start_time, 'sec'
        print '\tEnd   time =', self.spatiocyte_end_time, 'sec'
        print '\tInterval   =', self.spatiocyte_interval, 'sec'
	print '\tVoxel radius =', self.spatiocyte_VoxelRadius, 'm'
        print '\tCompartment lengths :', self.spatiocyte_lengths, 'voxels'
        print '\tSpecies Index :', self.spatiocyte_index
        print '\tObservable :', self.spatiocyte_observables


	# Visualization error	
	if self.spatiocyte_species_id is None:
	    raise VisualizerError('Cannot find species_id in any given csv files')

	if len(self.spatiocyte_data) == 0:
	    raise VisualizerError('Cannot find spatiocyte_data in any given csv files: ' \
	                    + ', '.join(csv_file_directry))

	if len(self.spatiocyte_index) == 0 :
	    raise VisualizerError('Cannot find spatiocyte_index in any given csv files: ' \
	                    + ', '.join(csv_file_directry))



class LSCMVisualizer(FCSVisualizer) :

	'''
	Confocal Visualization class of e-cell simulator
	'''

	def __init__(self, configs=LSCMConfigs(), effects=PhysicalEffects()) :

		assert isinstance(configs, LSCMConfigs)
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
		#	os.remove(os.path.join(self.configs.movie_image_file_dir, file))

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

		# photon flux density [#photons/(sec m2)]	
		n0 = self.configs.source_flux_density

		# scan time per pixel area
		dT = exposure_time/(Nh_pixel*Nw_pixel)

		# Cross-Section [m2]
		xsec = numpy.log(10)*(0.1*abs_coeff/Na)

		# the number of absorbed photons
		N_abs = xsec*n0*dT
	
		# Beer-Lambert law : A = log(I0/I) = coef * concentration * path-length
		A = (abs_coeff*0.1/Na)*(1/volume)*depth
	
		# the number of emitted photons
		N_emit0 = QY*N_abs*(1 - 10**(-A))

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

		# beam-position (nm) relative to cell-origin
		Y_b, Z_b = numpy.meshgrid(Np*(w-w0), Np*(h-h0))
		X_b = Nx*self.configs.detector_focal_point[0]

		# set flux density as a function of radial and depth
	        r = numpy.linspace(0, 20000, 20001)
	        d = numpy.linspace(0, 20000, 20001)

		# loop for scannig
                #count0 = int(start/exposure_time)
                #count0 = int(start_time/exposure_time)
		index0 = int(round(start/dt))
                delta_index = int(round(exposure_time/dt))

                while (time[-1] < end) :

		    count_imaging = int(round(time[0]/exposure_time))

		    print 'time : ', time[0], '-', time[-1], ' sec/image (', count_imaging, ')'

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

			time_bool = (time_index == index).astype('int')
			imaging = (time_bool*cell_bool).astype('int')
			len_imaging = len(imaging[imaging > 0])

			print index, index*dt+start, 'length :', len_imaging

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
			        R0 = dR.reshape((Nw_pixel*Nh_pixel))
			        R1 = (R0*imaging).astype('int')
			        radius = R1[R1 > 0]

				# set flux density as a function of radial and depth
				func_emit0 = interp1d(r, N_emit0[depth], bounds_error=False, fill_value=N_emit0[depth][r[-1]])

			        # the number of emitted photons
			        N_emit = func_emit0(radius)
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



#	def rewrite_InputData(self, output_file_dir=None) :
#
#		if not os.path.exists(output_file_dir):
#		    os.makedirs(output_file_dir)
#
#                # define observational image plane in nm-scale
#                voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9
#
#		# image dimenssion in pixel-scale
#		Nw_pixel = self.configs.detector_image_size[0]
#		Nh_pixel = self.configs.detector_image_size[1]
#
#                ## cell size (nm scale)
#                Nz = int(self.configs.spatiocyte_lengths[2]*voxel_size)
#                Ny = int(self.configs.spatiocyte_lengths[1]*voxel_size)
#                Nx = int(self.configs.spatiocyte_lengths[0]*voxel_size)
#
#		# pixel length : nm/pixel
#		Np = int(self.configs.image_scaling*voxel_size)
#
#		# image dimenssion in pixel-scale
#		Nz_pixel = Nz/Np
#		Ny_pixel = Ny/Np
#
#                # focal point
#                p_0 = numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point
#
#                # Beam position : 
#                beam_center = numpy.array(self.configs.detector_focal_point)
#
#		# exposure time
#		exposure_time = self.configs.detector_exposure_time
#
#		# contacting time with cells vertical-axis
#		R_z = float(Nz_pixel) / float(Nh_pixel)
#
#                z_exposure_time = exposure_time
#                z_contact_time  = R_z * z_exposure_time
#                z_nocontact_time = (z_exposure_time - z_contact_time)/2
#
#		# contacting time with cells horizontal-axis
#		R_y = float(Ny_pixel) / float(Nw_pixel)
#
#                y_exposure_time = z_exposure_time/Nw_pixel
#                y_contact_time  = R_y * y_exposure_time
#                y_nocontact_time = (y_exposure_time - y_contact_time)/2
#
#		# Active states per detection time
#		start = self.configs.spatiocyte_start_time
#		end   = self.configs.spatiocyte_end_time
#		dt = self.configs.spatiocyte_interval
#
#		# Abogadoro's number
#		Na = self.effects.avogadoros_number
#
#		# spatiocyte size
#		radius = self.configs.spatiocyte_VoxelRadius
#		volume = 4.0/3.0*numpy.pi*radius**3
#		depth  = 2.0*radius
#
#		# Quantum yield
#		QY = self.effects.quantum_yield
#
#		# Abs coefficient [1/(cm M)]
#		abs_coeff = self.effects.abs_coefficient
#
#		# photon flux density [#photons/(sec m2)]	
#		n0 = self.configs.source_flux_density
#
#		# scan time per pixel area
#		dT = exposure_time/(Nh_pixel*Nw_pixel)
#
#		# Cross-Section [m2]
#		xsec = numpy.log(10)*(0.1*abs_coeff/Na)
#
#		# the number of absorbed photons
#		N_abs = xsec*n0*dT
#	
#		# Beer-Lambert law : A = log(I0/I) = coef * concentration * path-length
#		A = (abs_coeff*0.1/Na)*(1/volume)*depth
#	
#		# the number of emitted photons
#		N_emit0 = QY*N_abs*(1 - 10**(-A))
#
#		# sequence
#		start_count = int(start/dt)
#		end_count = int(end/dt)
#
#		# copy input file
#		csv_input = self.configs.spatiocyte_file_directry + '/pt-input.csv'
#		shutil.copyfile(csv_input, output_file_dir + '/pt-input.csv')
#
#		# read input file
#		csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (0)
#		csv_list = list(csv.reader(open(csv_file_path, 'r')))
#
#		# fluorescence
#		if (self.effects.photobleaching_switch == True) :
#
#		    # get the number of particles
#		    N_particles = len(csv_list)
#
#		    # set fluorescence
#		    self.effects.set_photophysics_4lscm(start, end, dT, N_particles)
#
#		    bleach = self.effects.fluorescence_bleach
#		    budget = self.effects.fluorescence_budget*N_emit0[0][0]
#		    state  = self.effects.fluorescence_state
#
#		    scan_state = numpy.copy(state*0)
#		    scan_counter = numpy.zeros(shape=(N_particles))
#
#		# initialization
#		for count in range(start_count, end_count, 1) :
#
#		    csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (count)
#
#		    csv_list = list(csv.reader(open(csv_file_path, 'r')))
#		    dataset = numpy.array(csv_list)
#
#		    state_stack = numpy.column_stack((scan_state[:,count-start_count], (budget/N_emit0[0][0]).astype('int')))
#		    new_dataset = numpy.column_stack((dataset, state_stack))
#
#		    # write output file
#		    output_file = output_file_dir + '/pt-%09d.0.csv' % (count)
#
#		    with open(output_file, 'w') as f :
#		        writer = csv.writer(f)
#		        writer.writerows(new_dataset)
#
#
#		# loop for scannig
#		time = start
#
#                while (time < end) :
#
#		    print 'time : ', time, ' sec/image'
#
#		    # Beam position : initial
#		    p_b = numpy.array([Nx, Ny, Nz])*beam_center
#
#		    # contact time : z-direction
#		    z_scan_time = 0
#
#		    # no contact time : z-direction
#		    non_contact_time = z_nocontact_time
#
#		    # vertical-scanning sequences
#		    while (z_scan_time < z_contact_time) :
#
#		        # Beam position : z-direction
#		        beam_center[2] = z_scan_time/z_contact_time
#
#		        # contact time : y-direction
#		        y_scan_time = 0
#
#		        # no contact time : y-direction (left-margin)
#		        non_contact_time += y_nocontact_time
#
#		        # horizontal-scanning sequences
#		        while (y_scan_time < y_contact_time) :
#
#			    # Beam position : y-direction (cell in nm-scale)
#			    beam_center[1] = y_scan_time/y_contact_time
#
#			    p_b = numpy.array([Nx, Ny, Nz])*beam_center
#			    x_b, y_b, z_b = p_b
#
#			    # get scan-time and scan-count number
#			    scan_time  = z_scan_time + y_scan_time + non_contact_time
#
#			    # get image count number
#			    count = int((scan_time + time)/dt)
#
#		            # read input file
#		            csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (count)
#
#		            csv_list = list(csv.reader(open(csv_file_path, 'r')))
#		            dataset  = numpy.array(csv_list)
#
#                            # loop for particles
#                            for j, data_j in enumerate(dataset) :
#
#		                # set particle position
#		                c_id = (float(data_j[1]), float(data_j[2]), float(data_j[3]))
#		                x_i, y_i, z_i = self.get_coordinate(c_id)
#
#		                # beam axial position
#                                """ Distance between beam and fluorephore (depth) """
#		                d_s = abs(x_i - x_b)
#
#                                if (d_s < 20000) :
#		                    source_depth = d_s
#                                else :
#		                    source_depth = 19999
#
#		                # beam position
#                                """ Distance between beam position and fluorophore (plane) """
#		                rr = numpy.sqrt((y_i-y_b)**2 + (z_i-z_b)**2)
#
#		                if (rr < 20000) :
#		                    source_radius = rr
#		                else :
#		                    source_radius = 19999
#
#		                N_emit = N_emit0[int(source_depth)][int(source_radius)]
#
#				scan_counter[j] += (N_emit/N_emit0[0][0])
#				state_j = state[j,int(scan_counter[j])]
#
#		                photons = budget[j] - N_emit*state_j
#
#		                check = (photons > 0).astype('int')
#		                budget[j] = numpy.abs(photons*check)
#
#				scan_state[j,count-start_count] = state_j
#
#
#		            state_stack = numpy.column_stack((scan_state[:,count-start_count], (budget/N_emit0[0][0]).astype('int')))
#		            new_dataset = numpy.column_stack((dataset, state_stack))
#
#		            # write output file
#		            output_file = output_file_dir + '/pt-%09d.0.csv' % (count)
#
#		            with open(output_file, 'w') as f :
#		                writer = csv.writer(f)
#		                writer.writerows(new_dataset)
#
#
#			    y_scan_time += y_contact_time/Ny_pixel
#
#			# no contact time : y-direction (right-margin)
#			non_contact_time += y_nocontact_time
#
#			z_scan_time += z_contact_time/Nz_pixel
#
#		    time  += exposure_time
#
#
#
#
#	def get_molecule_plane(self, cell, data, pid, p_b, p_0) :
#
#		# get beam position
#		X_b, Y_b, Z_b = p_b
#
#		# particles coordinate, species and lattice IDs
#                c_id, s_id, l_id, p_state, cyc_id = data
#                
#		sid_array = numpy.array(self.configs.spatiocyte_species_id)
#		s_index = (numpy.abs(sid_array - int(s_id))).argmin()
#
#		if self.configs.spatiocyte_observables[s_index] is True :
#
#		    # particles coordinate in nm-scale
#                    p_i = self.get_coordinate(c_id)
#
#                    # get signal matrix
#                    #signal = self.get_signal(time, pid, s_index, p_i, p_b, p_0, p_state)
#                    signal = self.get_signal(p_i, p_b, p_0, p_state)
#
#                    # add signal matrix to image plane
#		    self.overwrite_signal(cell, signal, p_i, p_b)
#
#
#
        def get_signal(self, p_i, p_b, p_0, p_state, pinhole) :

		# set focal point
		x_0, y_0, z_0 = p_0

                # set source center
                x_b, y_b, z_b = p_b

		# set particle position
                x_i, y_i, z_i = p_i

		# image dimenssion in pixel-scale
		Nw_pixel = self.configs.detector_image_size[0]
		Nh_pixel = self.configs.detector_image_size[1]

		# scan time
		exposure_time = self.configs.detector_exposure_time
		dT = exposure_time/(Nw_pixel*Nh_pixel)

		# set arrays of radial and depth
		r = numpy.linspace(0, 20000, 20001)
		d = numpy.linspace(0, 20000, 20001)

		# axial position
                """ Distance between beam and fluorefore (depth) """
		dx = abs(x_b - x_i)
		depth = dx if dx < d[-1] else d[-1]

		# lateral position
                """ Distance between beam position and fluorophore (plane) """
		radius = numpy.sqrt((y_b - y_i)**2 + (z_b - z_i)**2)

		# get illumination PSF
		n0 = self.configs.source_flux_density

		# Absorption coeff [1/(cm M)]
		abs_coeff = self.effects.abs_coefficient

		# Quantum yield
		QY = self.effects.quantum_yield

		# Abogadoro's number
		Na = self.effects.avogadoros_number

		# Cross-section [m2]
		x_sec = numpy.log(10)*abs_coeff*0.1/Na

		# get the number of absorption photons : [#/(m2 sec)] [m2] [sec]
		N_abs = n0*x_sec*dT

		# spatiocyte size
		voxel_radius = self.configs.spatiocyte_VoxelRadius
		voxel_volume = (4.0/3.0)*numpy.pi*voxel_radius**3
		voxel_depth  = 2.0*voxel_radius

		# Beer-Lamberts law : log(I0/I) = A = abs coef. * concentration * path length ([m2] * [#/m3] * [m])
		A = (abs_coeff*0.1/Na)*(1.0/voxel_volume)*(voxel_depth)

		# get the number of photons emitted
		N_emit0 = p_state*QY*N_abs*(1 - 10**(-A))
		#func_emit = interp1d(r, N_emit0[depth], bounds_error=False, fill_value=N_emit0[depth][r[-1]])
		#func_emit = UnivariateSpline(r, N_emit0[depth])
		func_emit = InterpolatedUnivariateSpline(r, N_emit0[depth])

		N_emit = func_emit(radius)

		# get fluorophore PSF
		d_fluo = self.configs.depth
		dx = abs(x_i - x_0)
		depth_fluo = dx if dx < d_fluo[-1] else d_fluo[-1]

		fluo_psf = self.fluo_psf[int(depth_fluo)]

		# set a function of pinhole region cut
		Nr = len(pinhole)/2

		dy0, dz0 = (y_b - y_i) - Nr, (z_b - z_i) - Nr
		dy1, dz1 = (y_b - y_i) + Nr, (z_b - z_i) + Nr

		y_psf = numpy.linspace(fluo_psf[0,0], fluo_psf[-1,0], len(fluo_psf[:,0]))
		z_psf = numpy.linspace(fluo_psf[0,0], fluo_psf[0,-1], len(fluo_psf[0,:]))

		func_psf = interp2d(y_psf, z_psf, fluo_psf, bounds_error=False, fill_value=fluo_psf[-1,-1])
		#func_psf = RectBivariateSpline(y_psf, z_psf, fluo_psf)

		# set signal array
		dy = [numpy.linspace(dy0[k], dy1[k], 2*Nr) for k in range(len(radius))]
		dz = [numpy.linspace(dz0[k], dz1[k], 2*Nr) for k in range(len(radius))]

		signal  = numpy.array(map(lambda Y, Z, N : N*pinhole*func_psf(Y, Z), dy, dz, N_emit))
		photons = signal.sum(-1).sum(-1)

                return photons



#	def overwrite_signal(self, cell, signal, p_i, p_b) :
#
#		# particle position
#		x_i, y_i, z_i = p_i
#
#		# beam position
#		x_b, y_b, z_b = p_b
#
#		# z-axis
#		Nz_cell   = len(cell)
#		Nz_signal = len(signal)
#
#		Nh_cell = Nz_cell/2
#		Nh_signal = (Nz_signal-1)/2
#
#                z_to   = (z_i + Nh_signal) - (z_b - Nh_cell)
#                z_from = (z_i - Nh_signal) - (z_b - Nh_cell)
#
#		flag_z = True
#
#		if (z_to > Nz_cell + Nz_signal) :
#		    flag_z = False
#
#                elif (z_to > Nz_cell and
#		      z_to < Nz_cell + Nz_signal) :
#
#                    dz_to = z_to - Nz_cell
#
#                    zi_to = int(Nz_signal - dz_to)
#                    zb_to = int(Nz_cell)
#
#                elif (z_to > 0 and z_to < Nz_cell) :
#
#                    zi_to = int(Nz_signal)
#                    zb_to = int(z_to)
#
#		else : flag_z = False
#
#                if (z_from < 0) :
#
#                    zi_from = int(abs(z_from))
#                    zb_from = 0
#
#                else :
#
#                    zi_from = 0
#                    zb_from = int(z_from)
#
#		if (flag_z == True) :
#                    ddz = (zi_to - zi_from) - (zb_to - zb_from)
#
#                    if (ddz > 0) : zi_to = zi_to - ddz
#                    if (ddz < 0) : zb_to = zb_to + ddz
#
#                # y-axis
#                Ny_cell  = cell.size/Nz_cell
#                Ny_signal = signal.size/Nz_signal
#
#		Nh_cell = Ny_cell/2
#		Nh_signal = (Ny_signal-1)/2
#
#                y_to   = (y_i + Nh_signal) - (y_b - Nh_cell)
#                y_from = (y_i - Nh_signal) - (y_b - Nh_cell)
#
#		flag_y = True
#
#		if (y_to > Ny_cell + Ny_signal) :
#		    flag_y = False
#
#                elif (y_to > Ny_cell and
#		      y_to < Ny_cell + Ny_signal) :
#
#                    dy_to = y_to - Ny_cell
#
#                    yi_to = int(Ny_signal - dy_to)
#                    yb_to = int(Ny_cell)
#
#                elif (y_to > 0 and y_to < Ny_cell) :
#
#                    yi_to = int(Ny_signal)
#                    yb_to = int(y_to)
#
#		else : flag_y = False
#
#                if (y_from < 0) :
#
#                    yi_from = int(abs(y_from))
#                    yb_from = 0
#
#                else :
#
#                    yi_from = 0
#                    yb_from = int(y_from)
#
#		if (flag_y == True) :
#		    ddy = (yi_to - yi_from) - (yb_to - yb_from)
#
#		    if (ddy > 0) : yi_to = yi_to - ddy
#		    if (ddy < 0) : yb_to = yb_to + ddy
#
#		if (flag_z == True and flag_y == True) :
##		    if (abs(ddy) > 2 or abs(ddz) > 2) :
##		        print zi_from, zi_to, yi_from, yi_to, ddy, ddz
##		        print zb_from, zb_to, yb_from, yb_to
#
#		    # add to cellular plane
#                    cell[zb_from:zb_to, yb_from:yb_to] += signal[zi_from:zi_to, yi_from:yi_to]
#
#		#return cell
#
#
#
	def output_frames(self, num_div=1):

	    # set Fluorophores PSF
	    self.set_fluo_psf()

	    index_array = numpy.array(self.configs.shutter_index_array)
	    num_timesteps = len(index_array)
	    index0 = index_array[0]

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



        def output_frames_each_process(self, start_count, end_count):

		voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9

		# image dimenssion in pixel-scale
		Nw_pixel = self.configs.detector_image_size[0]
		Nh_pixel = self.configs.detector_image_size[1]

		# cells dimenssion in nm-scale
                Nz = int(self.configs.spatiocyte_lengths[2]*voxel_size)
                Ny = int(self.configs.spatiocyte_lengths[1]*voxel_size)
                Nx = int(self.configs.spatiocyte_lengths[0]*voxel_size)

		# pixel length : nm/pixel
		Np = int(self.configs.image_scaling*voxel_size)

		# cells dimenssion in pixel-scale
		Ny_pixel = Ny/Np
		Nz_pixel = Nz/Np

		# get focal point
		p_0 = self.get_focal_center()

		# Mesh-grid for pixel-image
		w = numpy.linspace(0, Nw_pixel-1, Nw_pixel)
		h = numpy.linspace(0, Nh_pixel-1, Nh_pixel)

		W, H = numpy.meshgrid(w, h)

                # exposure time
                exposure_time = self.configs.detector_exposure_time

                # spatiocyte time interval 
                data_interval = self.configs.spatiocyte_interval

		# set delta_index
                delta_index = int(round(exposure_time/data_interval))

		# scan-time (sec) per pixel-image
		dt_w = exposure_time/(Nh_pixel*Nw_pixel)
		dt_h = dt_w*Nw_pixel

		T_b = W*dt_w + H*dt_h
		time = exposure_time*start_count + T_b.reshape((Nw_pixel*Nh_pixel))

		# cell-center in pixel-image
		W0, H0 = Nw_pixel/2, Nh_pixel/2

		# cell in pixel-image
		w0, h0 = W0 - int(Ny_pixel/2), H0 - int(Nz_pixel/2)
		w1, h1 = w0 + Ny_pixel, w0 + Nz_pixel

		# cell-located region in pixel image
		cell = numpy.zeros((Nw_pixel, Nh_pixel))
		cell[w0:w1,h0:h1] = 1
		cell_bool = cell.reshape((Nw_pixel*Nh_pixel))

		# beam-position (nm) relative to cell-origin
		Y_b, Z_b = numpy.meshgrid(Np*(w-w0), Np*(h-h0))
		x_b = Nx*self.configs.detector_focal_point[0]

		# set radial and depth arrays
		r = self.configs.radial
		d = self.configs.depth

		# loop for scannig
		index_array = numpy.array(self.configs.shutter_count_array)
		index0 = index_array[0]

                if (time[-1] < exposure_time*end_count) :

		    # set count number
		    count_imaging = int(round(time[0]/exposure_time))

		    print 'time : ', time[0], '-', time[-1], ' sec/image (', count_imaging, ')'

                    image_file_name = os.path.join(self.configs.image_file_dir,
                                                self.configs.image_file_name_format % (count_imaging))

		    # set frame datasets
		    index1 = int(round(time[0]/data_interval))
		    index_start = (index1 - index0)
		    index_end   = (index1 - index0) + delta_index

                    frame_data = self.configs.spatiocyte_data[index_start:index_end]

		    print 'data length :', len(frame_data), 'index :', index0, index1, index_start, index_end

		    # set time_index
		    time_index = (time/data_interval).astype('int')

		    # define image in pixel-scale
		    image_pixel  = numpy.zeros(shape=(Nw_pixel, Nh_pixel, 2))
		    image_array0 = image_pixel[:,:,0].reshape((Nw_pixel*Nh_pixel))

		    if (len(frame_data) > 0) :

			# loop for scanning
		        for index in range(index_start, index_end, 1) :

			    # set cell-imaging region
			    time_bool = (time_index == index + index0).astype('int')
			    imaging = (time_bool*cell_bool).astype('int')

			    len_imaging = len(imaging[imaging > 0])

			    print index+index0, (index+index0)*data_interval, 'length :', len_imaging

			    if (len_imaging > 0) :

                        	# set frame_data
				i_time, i_data = frame_data[index-index_start]

			        # set beam position array
			        yy0 = Y_b.reshape((Nw_pixel*Nh_pixel))
			        y_b = yy0[imaging > 0]

			        zz0 = Z_b.reshape((Nw_pixel*Nh_pixel))
			        z_b = zz0[imaging > 0]

				# set the number of emitted photons
				photons = numpy.zeros(shape=(len_imaging))

				# set pinhole in nm-scale
				Mag = self.configs.image_magnification
				Nr  = int(self.configs.pinhole_radius/Mag/1e-9)

				pinhole = numpy.zeros(shape=(2*Nr,2*Nr))

				zz, yy = numpy.ogrid[-Nr:Nr,-Nr:Nr]
				rr_cut = yy**2 + zz**2 < Nr**2
				pinhole[rr_cut] = 1

                        	# loop for particles
                        	for j, data_j in enumerate(i_data) :

				    # particles coordinate, species and lattice IDs
                 		    c_id, m_id, s_id, l_id, p_state, cyc_id = data_j

				    sid_array = numpy.array(self.configs.spatiocyte_species_id)
				    s_index = (numpy.abs(sid_array - int(s_id))).argmin()

				    if self.configs.spatiocyte_observables[s_index] is True :

					if (p_state > 0) :
					    p_i = numpy.array(c_id)/1e-9

		        		    # particles coordinate in real(nm) scale
					    p_i, radial, depth = self.get_coordinate(p_i, p_0)
					    x_i, y_i, z_i = p_i

					    # set beam positional array in nm-scale
					    r_cut = numpy.sqrt((y_b - y_i)**2 + (z_b - z_i)**2) < 1.5*Nr
					    d_cut = abs(x_b - x_i) < 600 #d[-1]

					    p_b = (x_b, y_b[r_cut], z_b[r_cut])

					    if (len(y_b[r_cut]) > 0 and d_cut) :
                    			        # add photon signals per beam positional array
                    			        photons[r_cut] += self.get_signal(p_i, p_b, p_0, p_state, pinhole)

				# set the number of emitted photons to pixel-image
				image_array0[imaging > 0] = photons

			# get pixel-image
			image_pixel[:,:,0] = image_array0.reshape((Nw_pixel, Nh_pixel))
		        image = self.detector_output(image_pixel)

		        # save data to numpy-binary file
		        image_file_name = os.path.join(self.configs.image_file_dir,
					    self.configs.image_file_name_format % (count_imaging))
		        numpy.save(image_file_name, image)




#        def output_frames_each_process(self, start_count, stop_count):
#
#		voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9
#
#		# image dimenssion in pixel-scale
#		Nw_pixel = self.configs.detector_image_size[0]
#		Nh_pixel = self.configs.detector_image_size[1]
#
#		# cells dimenssion in nm-scale
#                Nz = int(self.configs.spatiocyte_lengths[2] * voxel_size)
#                Ny = int(self.configs.spatiocyte_lengths[1] * voxel_size)
#                Nx = int(self.configs.spatiocyte_lengths[0] * voxel_size)
#
#		# pixel length : nm/pixel
#		Np = int(self.configs.image_scaling*voxel_size)
#
#		# cells dimenssion in pixel-scale
#		Ny_pixel = Ny/Np
#		Nz_pixel = Nz/Np
#
#                # focal point
#                p_0 = numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point
#
#                # Beam position : 
#                beam_center = numpy.array(self.configs.detector_focal_point)
#
##                # set boundary condition
##                if (self.configs.spatiocyte_bc_switch == True) :
##
##                    bc = numpy.zeros(shape=(Nz, Ny))
##                    bc = self.set_boundary_plane(bc, p_b, p_0)
#
#		# exposure time
#		exposure_time = self.configs.detector_exposure_time
#
#		# contacting time with cells vertical-axis
#		R_z = float(Nz_pixel) / float(Nh_pixel)
#
#                z_exposure_time = exposure_time
#                z_contact_time  = R_z * z_exposure_time
#                z_nocontact_time = (z_exposure_time - z_contact_time)/2
#
#		# contacting time with cells horizontal-axis
#		R_y = float(Ny_pixel) / float(Nw_pixel)
#
#                y_exposure_time = z_exposure_time/Nw_pixel
#                y_contact_time  = R_y * y_exposure_time
#                y_nocontact_time = (y_exposure_time - y_contact_time)/2
#
#		#####
#                start_time = self.configs.spatiocyte_start_time
#
#                time = exposure_time * start_count
#                end  = exposure_time * stop_count
#
#                # data-time interval
#                data_interval = self.configs.spatiocyte_interval
#
#                delta_time = int(round(exposure_time / data_interval))
#
#                # create frame data composed by frame element data
#                count  = start_count
#                count0 = int(round(start_time / exposure_time))
#
#                # initialize Physical effects
#                #length0 = len(self.configs.spatiocyte_data[0][1])
#                #self.effects.set_states(t0, length0)
#
#                while (time < end) :
#
#                    image_file_name = os.path.join(self.configs.image_file_dir,
#                                        self.configs.image_file_name_format % (count))
#
#		    print 'time : ', time, ' sec (', count, ')'
#
#                    # define cell in nm-scale
#                    #cell = numpy.zeros(shape=(Nz, Ny))
#		    # define image array in pixel-scale
#		    image_pixel = numpy.zeros([Nw_pixel, Nh_pixel, 2])
#
#		    count_start = (count - count0)*delta_time
#		    count_end   = (count - count0 + 1)*delta_time
#
#                    frame_data = self.configs.spatiocyte_data[count_start:count_end]
#
#		    if (len(frame_data) > 0) :
#
#		        # Beam position : initial
#		        p_b = numpy.array([Nx, Ny, Nz])*beam_center
#
#			# Beam position : z-direction (image in pixel-scale)
#			z_image_pixel = int(z_nocontact_time/y_exposure_time)
#
#		        # contact time : z-direction
#		        z_scan_time = 0
#
#			# no contact time : z-direction
#			non_contact_time = z_nocontact_time
#
#			# vertical-scanning sequences
#		        while (z_scan_time < z_contact_time) :
#
#			    # Beam position : z-direction
#			    beam_center[2] = z_scan_time/z_contact_time
#
#			    # Beam position : y-direction (image in pixel-scale)
#			    y_image_pixel = int(y_nocontact_time/y_exposure_time*Nw_pixel)
#
#			    # contact time : y-direction
#			    y_scan_time = 0
#
#			    # no contact time : y-direction (left-margin)
#			    non_contact_time += y_nocontact_time
#
#			    # horizontal-scanning sequences
#			    while (y_scan_time < y_contact_time) :
#
#				# Beam position : y-direction (cell in nm-scale)
#				beam_center[1] = y_scan_time/y_contact_time
#
#				p_b = numpy.array([Nx, Ny, Nz])*beam_center
#				x_b, y_b, z_b = p_b
#
#                        	# loop for frame data
#				i_time, i_data = frame_data[0]
#
#				scan_time = z_scan_time + y_scan_time + non_contact_time
#				diff = abs(i_time - (scan_time + time))
#				data = i_data
#
#                        	for i, (i_time, i_data) in enumerate(frame_data) :
#                        	    #print '\t', '%02d-th frame : ' % (i), i_time, ' sec'
#				    i_diff = abs(i_time - (scan_time + time))
#
#				    if (i_diff < diff) :
#					diff = i_diff
#					data = i_data
#
#				# overwrite the scanned region to cell
#				#r_p = int(self.configs.image_scaling*voxel_size/2)
#				Mag = self.configs.image_magnification
#				r_p = int(self.configs.pinhole_radius/Mag/1e-9)
#
#				if (y_b-r_p < 0) : y_from = int(y_b)
#				else : y_from = int(y_b - r_p)
#
#				if (y_b+r_p >= Ny) : y_to = int(y_b)
#				else : y_to = int(y_b + r_p)
#
#				if (z_b-r_p < 0) : z_from = int(z_b)
#				else : z_from = int(z_b - r_p)
#
#				if (z_b+r_p >= Nz) : z_to = int(z_b)
#				else : z_to = int(z_b + r_p)
#
#				mask = numpy.zeros(shape=(z_to-z_from, y_to-y_from))
#
#				zz, yy = numpy.ogrid[z_from-int(z_b):z_to-int(z_b), y_from-int(y_b):y_to-int(y_b)]
#				rr_cut = yy**2 + zz**2 < r_p**2
#				mask[rr_cut] = 1
#
#				scan_cell = numpy.zeros_like(mask)
#
#                        	# loop for particles
#                        	for j, j_data in enumerate(data) :
#                        	    self.get_molecule_plane(scan_cell, i_time, j_data, j, p_b, p_0)
#
#				# image pixel position
#				ii, jj = z_image_pixel, y_image_pixel
#
#				#cell[z_from:z_to, y_from:y_to] += mask*scan_cell
#				Photon_flux = numpy.sum(mask*scan_cell)
#				image_pixel[ii][jj][0] = Photon_flux
#
#				y_scan_time += y_contact_time/Ny_pixel
#				y_image_pixel += 1
#
#			    # no contact time : y-direction (right-margin)
#			    non_contact_time += y_nocontact_time
#			    y_image_pixel += int(y_nocontact_time/y_exposure_time*Nw_pixel)
#
#			    z_scan_time += z_contact_time/Nz_pixel
#			    z_image_pixel += 1
#
#
#		    if (numpy.amax(image_pixel) > 0) :
#
#		        image = self.detector_output(image_pixel)
#
#		        # save data to numpy-binary file
#		        image_file_name = os.path.join(self.configs.image_file_dir,
#					self.configs.image_file_name_format % (count))
#		        numpy.save(image_file_name, image)
#
#		    	# save data to png-image file
#		    	#image[:,:,3].astype('uint%d' % (self.configs.ADConverter_bit))
#		    	#toimage(image[:,:,3], low=numpy.amin(image[:,:,3]), high=numpy.amax(image[:,:,3]), mode='I').save(image_file_name)
#
#		    time  += exposure_time
#		    count += 1
#
#
#
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




	#def detector_output(self, cell) :
	def detector_output(self, image_pixel) :

		# Detector Output
		voxel_radius = self.configs.spatiocyte_VoxelRadius
                voxel_size = (2.0*voxel_radius)/1e-9

		Nw_pixel = self.configs.detector_image_size[0]
		Nh_pixel = self.configs.detector_image_size[1]

#		Np = int(self.configs.image_scaling*voxel_size)
#
#                # image in nm-scale
#		Nw_image = Nw_pixel*Np
#		Nh_image = Nh_pixel*Np
#
#		Nw_cell = len(cell)
#		Nh_cell = len(cell[0])
#
#		if (Nw_image > Nw_cell) :
#
#		    w_cam_from = int((Nw_image - Nw_cell)/2.0)
#		    w_cam_to   = w_cam_from + Nw_cell
#                    w_cel_from = 0
#                    w_cel_to   = Nw_cell
#
#		else :
#
#                    w_cam_from = 0
#                    w_cam_to   = Nw_image
#                    w_cel_from = int((Nw_cell - Nw_image)/2.0)
#                    w_cel_to   = w_cel_from + Nw_image
#
#		if (Nh_image > Nh_cell) :
#
#                    h_cam_from = int((Nh_image - Nh_cell)/2.0)
#                    h_cam_to   = h_cam_from + Nh_cell
#                    h_cel_from = 0
#                    h_cel_to   = Nh_cell
#
#                else :
#
#                    h_cam_from = 0
#                    h_cam_to   = int(Nh_image)
#                    h_cel_from = int((Nh_cell - Nh_image)/2.0)
#                    h_cel_to   = h_cel_from + Nh_image
#
#
#		# image in nm-scale
#		plane = cell[w_cel_from:w_cel_to, h_cel_from:h_cel_to]
#
#		# convert image in nm-scale to pixel-scale
#                cell_pixel = numpy.zeros(shape=(Nw_cell/Np, Nh_cell/Np))
#
#		# Signal (photon distribution on cell)
#		for i in range(Nw_cell/Np) :
#		    for j in range(Nh_cell/Np) :
#
#			# get photon flux
#			Photon_flux = numpy.sum(plane[i*Np:(i+1)*Np, j*Np:(j+1)*Np])
#			cell_pixel[i][j] = Photon_flux
#
#
#                # Background (photon distribution on image)
#                image_pixel = numpy.zeros([Nw_pixel, Nh_pixel, 2])
#
#		w_cam_from = int(w_cam_from/Np)
#		w_cam_to   = int(w_cam_to/Np)
#		h_cam_from = int(h_cam_from/Np)
#		h_cam_to   = int(h_cam_to/Np)
#
#		w_cel_from = int(w_cel_from/Np)
#		w_cel_to   = int(w_cel_to/Np)
#		h_cel_from = int(h_cel_from/Np)
#		h_cel_to   = int(h_cel_to/Np)
#
#		ddw = (w_cam_to - w_cam_from) - (w_cel_to - w_cel_from)
#		ddh = (h_cam_to - h_cam_from) - (h_cel_to - h_cel_from)
#
#		if   (ddw > 0) : w_cam_to = w_cam_to - ddw
#		elif (ddw < 0) : w_cel_to = w_cel_to - ddw
#
#                if   (ddh > 0) : h_cam_to = h_cam_to - ddh
#                elif (ddh < 0) : h_cel_to = h_cel_to - ddh
#
#		# place cell_pixel data to image image
#		image_pixel[w_cam_from:w_cam_to, h_cam_from:h_cam_to, 0] = cell_pixel[w_cel_from:w_cel_to, h_cel_from:h_cel_to]

		# set seed for random number
		numpy.random.seed()

		# observational time
		dT = self.configs.detector_exposure_time/(Nw_pixel*Nh_pixel)

		# conversion : photon --> photoelectron --> ADC count
                for i in range(Nw_pixel) :
                    for j in range(Nh_pixel) :

			# pixel position
			pixel = (i, j)

			# Detector : Quantum Efficiency
			#index = int(self.configs.psf_wavelength) - int(self.configs.wave_length[0])
			QE = self.configs.detector_qeff

                        # get signal (photon flux and photons)
			Photons = image_pixel[i][j][0]
			Flux = Photons/dT

			if (self.configs.detector_mode == "Photon-counting") :
			    # pair-pulses time resolution (sec)
			    t_pp = self.configs.detector_pair_pulses
			    Flux = Flux/(1 + Flux*t_pp)

                        # get constant background
                        if (self.effects.background_switch == True) :

                            Photons_bg = self.effects.background_mean
                            Photons += Photons_bg

			# get signal (expectation)
			Exp = QE*Photons

			# get dark count
			D = self.configs.detector_dark_count
			Exp += D*dT

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

			# A/D converter : Expectation --> ADC counts
			EXP = self.get_ADC_value(pixel, Exp+Nr)

			# A/D converter : Photoelectrons --> ADC counts
			PE  = signal + noise
			ADC = self.get_ADC_value(pixel, PE)

			# set data in image array
			#image_pixel[i][j] = [Photons, Exp, PE, ADC]
			image_pixel[i][j] = [EXP, ADC]

		return image_pixel


