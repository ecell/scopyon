import sys
import os
import shutil
import copy
import tempfile
import time
import math
import operator
import random
import csv
#import h5py

import scipy
import numpy

import palm_configs
from tirfm_handler import VisualizerError, TIRFMConfigs, TIRFMVisualizer
from effects_handler import PhysicalEffects

from scipy.special import j0
from scipy.misc    import toimage


class PALMConfigs(TIRFMConfigs) :

    '''

    PALM configration

	TIRFMiroscopy
	     +
	Photo-bleaching/-blinking
    '''

    def __init__(self, user_configs_dict = None):

        # default setting
        configs_dict = palm_configs.__dict__.copy()
        #configs_dict = parameter_configs.__dict__.copy()
        #configs_dict_tirfm = tirfm_configs.__dict__.copy()
        #configs_dict.update(configs_dict_tirfm)

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



    def set_ExcitationSource(self,  source_type = None,
				wave_length = None,
                                flux_density = None,
                                radius = None,
                                angle  = None ) :

        self._set_data('source_excitation_switch', True)
        self._set_data('source_excitation_type', source_type)
        self._set_data('source_excitation_wavelength', wave_length)
        self._set_data('source_excitation_flux_density', flux_density)
        self._set_data('source_excitation_radius', radius)
        self._set_data('source_excitation_angle', angle)

        print '--- Excitation Source :', self.source_excitation_type
        print '\tWave Length = ', self.source_excitation_wavelength, 'nm'
        print '\tFlux Density = ', self.source_excitation_flux_density, 'W/cm2'
        print '\t1/e2 Radius = ', self.source_excitation_radius, 'm'
        print '\tAngle = ', self.source_excitation_angle, 'degree'




    def set_ActivationSource(self,  source_type = None,
				wave_length = None,
                                flux_density = None,
                                radius = None,
                                angle  = None,
                                frame_time = None,
                                F_bleach = None) :

        self._set_data('source_activation_switch', True)
        self._set_data('source_activation_type', source_type)
        self._set_data('source_activation_wavelength', wave_length)
        self._set_data('source_activation_flux_density', flux_density)
        self._set_data('source_activation_radius', radius)
        self._set_data('source_activation_angle', angle)
        self._set_data('source_activation_frame_time', frame_time)
        self._set_data('source_activation_bleaching_frames', F_bleach)

        print '--- Activation Source :', self.source_activation_type
        print '\tWave Length = ', self.source_activation_wavelength, 'nm'
        print '\tFlux Density = ', self.source_activation_flux_density, 'W/cm2'
        print '\t1/e2 Radius = ', self.source_activation_radius, 'm'
        print '\tAngle = ', self.source_activation_angle, 'degree'
	print '\t'
        print '\tFrame time = ', self.source_activation_frame_time, 'sec'
        print '\tBleaching frames = ', self.source_activation_bleaching_frames




    def set_Excitation_path(self) :

        #r = self.radial
        #d = self.depth
	r = numpy.linspace(0, 20000, 20001)
	d = numpy.linspace(0, 20000, 20001)

        # (plank const) * (speed of light) [joules meter]
        hc = self.hc_const

	# Illumination : Assume that uniform illumination (No gaussian)
	# flux density [W/cm2 (joules/sec/cm2)]
        P_0 = self.source_excitation_flux_density*1e+4

        # single photon energy
        wave_length = self.source_excitation_wavelength*1e-9
        E_wl = hc/wave_length

	# photon flux density [photons/sec/cm2]
        N_0 = P_0/E_wl

	################################################################
	# Evanescent field
	#
	#	Assume uniform beam profile (Not Gaussian)
	#		Linear polarization
	#
        ################################################################

	# Incident beam : Angle
	theta_in = (self.source_excitation_angle/180.)*numpy.pi

	sin  = numpy.sin(theta_in)
	cos  = numpy.cos(theta_in)
	sin2 = sin**2
	cos2 = cos**2

	# index refraction
	n_1 = 1.46 # fused silica
	n_2 = 1.33 # water (objective : water immersion)
	n  = n_2/n_1 # must be < 1
	n2 = n**2

	# Incident beam : Amplitude
	#theta = numpy.pi/2.0
	A2_Is = N_0#*numpy.cos(theta)**2
	A2_Ip = N_0#*numpy.sin(theta)**2

	if (sin2/n2 > 1) :
	    # Evanescent field : Amplitude and Depth
	    # Assume that the s-polar direction is parallel to y-axis
	    A2_x = A2_Ip*(4*cos2*(sin2 - n2)/(n2**2*cos2 + sin2 - n2))
	    A2_y = A2_Is*(4*cos2/(1 - n2))
	    A2_z = A2_Ip*(4*cos2*sin2/(n2**2*cos2 + sin2 - n2))

	    A2_Tp = A2_x + A2_z
	    A2_Ts = A2_y

	    depth = wave_length/(4.0*numpy.pi*numpy.sqrt(n_1**2*sin2 - n_2**2))

	else :
	    # Epi-fluorescence field : Amplitude and Depth
	    cosT = numpy.sqrt(1 - sin2/n2)

	    A2_Tp = A2_Ip*(2*cos/(cosT + n*cos))**2
	    A2_Ts = A2_Is*(2*cos/(n*cosT + cos))**2

	    depth = float('inf')

	I_d = numpy.exp(-d*1e-9/depth)
	I_r = numpy.array(map(lambda x : A2_Tp+A2_Ts, r*1e-9))

	# photon flux density [photon/(sec m^2)]
        self.source_excitation_flux_density = numpy.array(map(lambda x : I_r*x, I_d))

	print 'Penetration depth :', depth, 'm'
	print 'Photon Flux Density (Max) :', self.source_excitation_flux_density[0][0], '#photon/(sec m^2)'



    def set_Activation_path(self) :

        #r = self.radial
        #d = self.depth
	r = numpy.linspace(0, 20000, 20001)
	d = numpy.linspace(0, 20000, 20001)

        # (plank const) * (speed of light) [joules meter]
        hc = self.hc_const

	# Illumination : Assume that uniform illumination (No gaussian)
	# flux density [W/cm2 (joules/sec/cm2)]
        P_0 = self.source_activation_flux_density*1e+4

        # single photon energy
        wave_length = self.source_activation_wavelength*1e-9
        E_wl = hc/wave_length

	# photon flux density [photons/sec/cm2]
        N_0 = P_0/E_wl

	################################################################
	# Evanescent field
	#
	#	Assume uniform beam profile (Not Gaussian)
	#		Linear polarization
	#
        ################################################################

	# Incident beam : Angle
	theta_in = (self.source_activation_angle/180.)*numpy.pi

	sin  = numpy.sin(theta_in)
	cos  = numpy.cos(theta_in)
	sin2 = sin**2
	cos2 = cos**2

	# index refraction
	n_1 = 1.46 # fused silica
	n_2 = 1.33 # water (objective : water immersion)
	n  = n_2/n_1 # must be < 1
	n2 = n**2

	# Incident beam : Amplitude
	#theta = numpy.pi/2.0
	A2_Is = N_0#*numpy.cos(theta)**2
	A2_Ip = N_0#*numpy.sin(theta)**2

	if (sin2/n2 > 1) :
	    # Evanescent field : Amplitude and Depth
	    # Assume that the s-polar direction is parallel to y-axis
	    A2_x = A2_Ip*(4*cos2*(sin2 - n2)/(n2**2*cos2 + sin2 - n2))
	    A2_y = A2_Is*(4*cos2/(1 - n2))
	    A2_z = A2_Ip*(4*cos2*sin2/(n2**2*cos2 + sin2 - n2))

	    A2_Tp = A2_x + A2_z
	    A2_Ts = A2_y

	    depth = wave_length/(4.0*numpy.pi*numpy.sqrt(n_1**2*sin2 - n_2**2))

	else :
	    # Epi-fluorescence field : Amplitude and Depth
	    cosT = numpy.sqrt(1 - sin2/n2)

	    A2_Tp = A2_Ip*(2*cos/(cosT + n*cos))**2
	    A2_Ts = A2_Is*(2*cos/(n*cosT + cos))**2

	    depth = float('inf')

	I_d = numpy.exp(-d*1e-9/depth)
	I_r = numpy.array(map(lambda x : A2_Tp+A2_Ts, r*1e-9))

	# photon flux density [photon/(sec m^2)]
        self.source_activation_flux_density = numpy.array(map(lambda x : I_r*x, I_d))

	print 'Penetration depth :', depth, 'm'
	print 'Photon Flux Density (Max) :', self.source_activation_flux_density[0][0], '#photon/(sec m^2)'



    def set_Optical_path(self) :

	# (1) Illumination path : Light source --> Sample
	self.set_Excitation_path()
	self.set_Activation_path()

	# (2) Detection path : Sample --> Detector
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



class PALMVisualizer(TIRFMVisualizer) :

	'''
	PALM visualization class
	'''

	def __init__(self, configs=PALMConfigs(), effects=PhysicalEffects()) :

		assert isinstance(configs, PALMConfigs)
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
                set optical path from source to detector
                """
		self.configs.set_Optical_path()




	def rewrite_InputData(self, output_file_dir=None) :

		if not os.path.exists(output_file_dir):
		    os.makedirs(output_file_dir)

                # define observational image plane in nm-scale
                voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9

                ## cell size (nm scale)
                Nz = int(self.configs.spatiocyte_lengths[2]*voxel_size)
                Ny = int(self.configs.spatiocyte_lengths[1]*voxel_size)
                Nx = int(self.configs.spatiocyte_lengths[0]*voxel_size)

                # focal point
                p_0 = numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point

                # beam position : Assuming beam position = focal point for temporary
                x_b, y_b, z_b = copy.copy(p_0)

		# Active states per detection time
		start = self.configs.spatiocyte_start_time
		end   = self.configs.spatiocyte_end_time
		dt = self.configs.spatiocyte_interval

		tau_frame = self.configs.source_activation_frame_time
		F_bleach  = self.configs.source_activation_bleaching_frames

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
		#n0 = self.configs.source_excitation_flux_density[0][0]
		n0 = self.configs.source_excitation_flux_density

		# Cross-Section [m2]
		xsec = numpy.log(10)*(0.1*abs_coeff/Na)

		# the number of absorbed photons
		N_abs = xsec*n0*dt
	
		# Beer-Lambert law : A = log(I0/I) = coef * concentration * path-length
		A = (abs_coeff*0.1/Na)*(1/volume)*depth
	
		# the number of emitted photons
		N_emit0 = QY*N_abs*(1 - 10**(-A))

		# sequence
		start_count = int(start/dt)
		end_count = int(end/dt)

		# copy input file
		csv_input = self.configs.spatiocyte_file_directry + '/pt-input.csv'
		shutil.copyfile(csv_input, output_file_dir + '/pt-input.csv')

		# read input file
		csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (0)
		csv_list = list(csv.reader(open(csv_file_path, 'r')))
		#csv_temp = list(csv.reader(open(csv_file_path, 'r')))
		#csv_list = [csv_temp[0]]

		# fluorescence
		if (self.effects.photobleaching_switch == True) :

		    # activation frames
		    tau_frame = self.configs.source_activation_frame_time
		    f = int(tau_frame/dt)

		    # bleaching frames
		    F_bleach  = self.configs.source_activation_bleaching_frames
		    exposure_time = self.configs.detector_exposure_time
		    F = int((F_bleach*exposure_time)/dt)

		    # get the number of particles
		    N_particles = len(csv_list)

		    # set fluorescence
		    self.effects.set_photophysics_4palm(start, end, dt, f, F, N_particles)

		    bleach = self.effects.fluorescence_bleach
		    budget = self.effects.fluorescence_budget*N_emit0[0][0]
		    state  = self.effects.fluorescence_state


		for count in range(start_count, end_count, 1) :

		    # read input file
		    csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (count)

		    csv_list = list(csv.reader(open(csv_file_path, 'r')))
		    #csv_temp = list(csv.reader(open(csv_file_path, 'r')))
		    #csv_list = [csv_temp[0]]
		    dataset = numpy.array(csv_list)

		    time = float(dataset[count][0])

                    # loop for particles
                    for j, data_j in enumerate(dataset):
		        # set particle position
		        c_id = (float(data_j[1]), float(data_j[2]), float(data_j[3]))
		        x_i, y_i, z_i = self.get_coordinate(c_id)

		        r = self.configs.radial
		        d = self.configs.depth

		        # beam axial position
                        """ Distance between beam and fluorephore (depth) """
		        d_s = abs(x_i - x_b)

                        if (d_s < 20000) :
		            source_depth = d_s
                        else :
		            source_depth = 19999

		        # beam position
                        """ Distance between beam position and fluorophore (plane) """
		        rr = numpy.sqrt((y_i-y_b)**2 + (z_i-z_b)**2)

		        if (rr < 20000) :
		            source_radius = rr
		        else :
		            source_radius = 19999

		        N_emit = N_emit0[int(source_depth)][int(source_radius)]*state[j,count-start_count]
		        photons = budget[j] - N_emit

		        check  = (photons > 0).astype('int')
		        budget[j] = numpy.abs(photons*check)

		    state_stack = numpy.column_stack((state[:,count-start_count], (budget/N_emit0[0][0]).astype('int')))
		    new_dataset = numpy.column_stack((dataset, state_stack))

		    #print float(new_dataset[0][0]), N_emit[0], budget[0], state[0,count], int(budget[0]/N_emit0[0][0])

		    # write output file
		    output_file = output_file_dir + '/pt-%09d.0.csv' % (count)

		    with open(output_file, 'w') as f :
		        writer = csv.writer(f)
		        writer.writerows(new_dataset)

#		    for i, dataset in enumerate(csv.reader(open(csv_file_path, 'r'))) :
#
#		        time = float(dataset[0])
#
#		        # get current index of fluorescence-state array
#		        k = (numpy.abs(time_act[i] - time)).argmin()
#
#		        # Photo-activation (ON-state)
#		        if (state_act[i][k] == 1) :
#			    N_emit = N_emit0
#	
#		        # Photo-activation (OFF-state)
#		        else : N_emit = 0
#
#		        if (budget[i] > 0) :
#		            budget[i] = budget[i] - N_emit
#		        else :
#			    Budget = 0
#			    N_emit = 0
#
#			state = (int(N_emit/N_emit0),int(budget[i]/N_emit0))
#
#			dataset.append(state)
#
#			with open(output_file, 'a') as f :
#			    writer = csv.writer(f)
#			    writer.writerow(dataset)




	def get_molecule_plane(self, cell, time, data, pid, p_b, p_0) :

		voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9
                
		# particles coordinate, species and lattice IDs
                c_id, s_id, l_id, p_state, cyc_id = data
                
		sid_array = numpy.array(self.configs.spatiocyte_species_id)
		s_index = (numpy.abs(sid_array - int(s_id))).argmin()

		if self.configs.spatiocyte_observables[s_index] is True :

		    # particles coordinate in real(nm) scale
                    #pos = self.get_coordinate(c_id)
                    #p_i = numpy.array(pos)*voxel_size
                    p_i = self.get_coordinate(c_id)

                    # get signal matrix
                    signal = self.get_signal(time, pid, s_index, p_i, p_b, p_0, p_state)

                    # add signal matrix to image plane
                    self.overwrite_signal(cell, signal, p_i)



        def get_signal(self, time, pid, s_index, p_i, p_b, p_0, p_state) :

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
                """ Distance between beam and fluorefore (depth) """
		d_s = abs(x_i - x_b)

                if (d_s < 20000) :
		    source_depth = d_s
                else :
		    source_depth = 19999

		# beam position
                """ Distance between beam position and fluorophore (plane) """
		rr = numpy.sqrt((y_i-y_b)**2 + (z_i-z_b)**2)

		if (rr < 20000) :
		    source_radius = rr
		else :
		    source_radius = 19999

		# get illumination PSF
		source_psf = self.configs.source_excitation_flux_density[int(source_depth)][int(source_radius)]

		# fluorophore axial position
		d_f = abs(x_i - x_b)

		if (d_f < len(d)) :
		    fluo_depth = d_f
		else :
		    fluo_depth = d[-1]

		# get fluorophore PSF
		fluo_psf = self.fluo_psf[int(fluo_depth)]

		# Spatiocyte time interval [sec]
		unit_time = self.configs.spatiocyte_interval

		# Absorption coeff [1/(cm M)]
		abs_coeff = self.effects.abs_coefficient

		# Quantum yield
		QY = self.effects.quantum_yield

		# Abogadoro's number
		Na = self.effects.avogadoros_number

		# Cross-section [m2]
		x_sec = numpy.log(10)*abs_coeff*0.1/Na

		# get the number of absorption photons : [#/(m2 sec)] [m2] [sec]
		n_abs = source_psf*x_sec*unit_time

		# spatiocyte size
		voxel_radius = self.configs.spatiocyte_VoxelRadius
		voxel_volume = (4.0/3.0)*numpy.pi*voxel_radius**3
		voxel_depth  = 2.0*voxel_radius

		# Beer-Lamberts law : log(I0/I) = A = abs coef. * concentration * path length ([m2] * [#/m3] * [m])
		A = (abs_coeff*0.1/Na)*(1.0/voxel_volume)*(voxel_depth)

		# get the number of photons emitted
		N_emit = p_state*QY*n_abs*(1 - 10**(-A))

		# get signal
		signal = N_emit/(4.0*numpy.pi) * fluo_psf

                return signal



