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

import parameter_configs
from tirfm_handler import VisualizerError, TIRFMConfigs, TIRFMVisualizer
from effects_handler import PhysicalEffects

from scipy.special import j0
from scipy.misc    import toimage


class MorigaConfigs(TIRFMConfigs) :

    '''

    Moriga-TIRFM configration

	Evanescent Field
	    +
	Detector : EMCCD/CMOS
    '''

    def __init__(self, user_configs_dict = None):

        # default setting
        configs_dict = parameter_configs.__dict__.copy()
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



    def set_Fluorophore(self, fluorophore_type = None,
                              wave_length = None,
			      intensity = None,
			      noise = None,
                              width = None,
                              cutoff = None,
                              file_name_format = None) :

        print '--- Fluorophore : Point Spreading Function [Gaussian]'

        self._set_data('psf_wavelength', wave_length)
        self._set_data('psf_intensity', intensity)
        #self._set_data('psf_intensity_noise', noise)
        self._set_data('psf_width', width)
        self._set_data('psf_cutoff', cutoff)
        self._set_data('psf_file_name_format', file_name_format)

	index = (numpy.abs(self.wave_length - self.psf_wavelength)).argmin()

	self.fluoex_eff[index] = 100
	self.fluoem_eff[index] = 100

        print '\tWave Length   = ', self.psf_wavelength, 'nm'
	print '\tIntensity = ', self.psf_intensity
	#print '\tIntensity Noise = ', self.psf_intensity_noise
        print '\tLateral Width = ', self.psf_width[0], 'nm'
        print '\tAxial Width = ', self.psf_width[1], 'nm'
	print '\tLateral Cutoff = ', self.psf_cutoff[0], 'nm'
        print '\tAxial Cutoff = ', self.psf_cutoff[1], 'nm'
                        
	# Normalization
	norm = sum(self.fluoex_eff)
	self.fluoex_norm = numpy.array(self.fluoex_eff)/norm

	norm = sum(self.fluoem_eff)
	self.fluoem_norm = numpy.array(self.fluoem_eff)/norm



    def set_PSF_detector(self) :

        r = self.radial
        z = self.depth

        wave_length = self.psf_wavelength

        # Fluorophores Emission Intensity (wave_length)
        I = self.fluoem_norm

        # Photon Transmission Efficiency
        if (self.dichroic_switch == True) :
            I = I*0.01*self.dichroic_eff

        if (self.emission_switch == True) :
            I = I*0.01*self.emission_eff

	# For normalization
	norm = map(lambda x : True if x > 1e-4 else False, I)

	# PSF : Fluorophore
        I0 = 1.0#self.psf_intensity
        Ir = sum(map(lambda x : x*numpy.exp(-0.5*(r/self.psf_width[0])**2), norm))
        Iz = sum(map(lambda x : x*numpy.exp(-0.5*(z/self.psf_width[1])**2), norm))

	psf_fl = numpy.array(map(lambda x : I0*Ir*x, Iz))

	self.fluorophore_psf = psf_fl



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
		        #intensity = float(data_id[7])
		        #intensity = (0.458/0.75)*self.psf_intensity

		        I0 = (0.458/0.75)*self.psf_intensity
			I = numpy.array([0.1*i for i in range(2000)])
			A = 0.025; w0 = 14; w1 = 9
			ext_func = A*numpy.exp(-numpy.exp(-(I-I0)/w1) - ((I-I0)/w0) + 1)
			ext_sum  = ext_func.sum()
			intensity = numpy.random.choice(I, None, p=ext_func/ext_sum)

		    except Exception :
			print 'Error : Missing intensity'

		    particles.append((c_id, s_id, l_id, intensity))

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




class MorigaVisualizer(TIRFMVisualizer) :

	'''
	Moriga-TIRFM visualization class
	'''

	def __init__(self, configs=MorigaConfigs(), effects=PhysicalEffects()) :

		assert isinstance(configs, MorigaConfigs)
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
		self.configs.set_Detection_path()



	def rewrite_InputData(self, output_file_dir=None) :

		if not os.path.exists(output_file_dir):
		    os.makedirs(output_file_dir)

		# Active states per detection time
		start = self.configs.spatiocyte_start_time
		end   = self.configs.spatiocyte_end_time
		dt = self.configs.spatiocyte_interval

		# spatiocyte size
		radius = self.configs.spatiocyte_VoxelRadius
		volume = 4.0/3.0*numpy.pi*radius**3
		depth  = 2.0*radius

		# set signal probability function (PDF)
		I0 = 46.7
		A = 0.0274
		w0 = 12.2
		w1 = 9.40
		I = numpy.array([0.01*i for i in range(2000000)])
		p_signal = A*numpy.exp(-numpy.exp(-((I-I0)/w1))-((I-I0)/w0)+1)
		p_sum = p_signal.sum()

		# sequence
		start_count = int(start/dt)
		end_count = int(end/dt)

		# copy input file
		csv_input = self.configs.spatiocyte_file_directry + '/pt-input.csv'
		shutil.copyfile(csv_input, output_file_dir + '/pt-input.csv')

		for count in range(start_count, end_count, 1) :

		    # read input file
		    csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (count)

		    csv_list = list(csv.reader(open(csv_file_path, 'r')))
		    #csv_temp = list(csv.reader(open(csv_file_path, 'r')))
		    #csv_list = [csv_temp[0]]
		    dataset = numpy.array(csv_list)

		    time = float(dataset[0][0])

		    N_particles = len(csv_list)

		    I0 = self.configs.psf_intensity
		    Noise = self.configs.psf_intensity_noise
		    photons = numpy.random.normal(I0, Noise, N_particles)
		    signals = numpy.random.choice(I, N_particles, p=p_signal/p_sum)
		    photons = (0.458/0.75)*signals

		    check  = (photons > 0).astype('int')
		    Intensity = numpy.abs(photons*check)

		    new_dataset = numpy.column_stack((dataset, Intensity))

		    #print float(new_dataset[0][0]), photons

		    # write output file
		    output_file = output_file_dir + '/pt-%09d.0.csv' % (count)

		    with open(output_file, 'w') as f :
		        writer = csv.writer(f)
		        writer.writerows(new_dataset)



	def get_molecule_plane(self, cell, time, data, pid, p_b, p_0) :

		voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9
                
		# particles coordinate, species and lattice IDs
                c_id, s_id, l_id, intensity = data
                
		sid_array = numpy.array(self.configs.spatiocyte_species_id)
		s_index = (numpy.abs(sid_array - int(s_id))).argmin()

		if self.configs.spatiocyte_observables[s_index] is True :

		    # particles coordinate in real(nm) scale
                    #pos = self.get_coordinate(c_id)
                    #p_i = numpy.array(pos)*voxel_size
                    p_i = self.get_coordinate(c_id)

                    # get signal matrix
                    signal = self.get_signal(time, pid, s_index, p_i, p_b, p_0, intensity)

                    # add signal matrix to image plane
                    self.overwrite_signal(cell, signal, p_i)



        def get_signal(self, time, pid, s_index, p_i, p_b, p_0, intensity) :

		# set focal point
		x_0, y_0, z_0 = p_0

                # set source center
                x_b, y_b, z_b = p_b

		# set particle position
                x_i, y_i, z_i = p_i

		#
		r = self.configs.radial
		d = self.configs.depth

		# fluorophore axial position
		d_f = abs(x_i - x_b)

		if (d_f < len(d)) :
		    fluo_depth = d_f
		else :
		    fluo_depth = d[-1]

		# get fluorophore PSF
		fluo_psf = self.fluo_psf[int(fluo_depth)]

		#
		voxel_radius = self.configs.spatiocyte_VoxelRadius
		voxel_size = (2.0*voxel_radius)/1e-9
		Np = int(self.configs.image_scaling*voxel_size)

		Norm = intensity/(Np**2)

		# get signal
		#signal = fluo_psf
		signal = Norm*fluo_psf

                return signal



        def output_frames_each_process(self, start_count, stop_count):

                # define observational image plane in nm-scale
                voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9

                ## cell size (nm scale)
                Nz = int(self.configs.spatiocyte_lengths[2]*voxel_size)
                Ny = int(self.configs.spatiocyte_lengths[1]*voxel_size)
                Nx = int(self.configs.spatiocyte_lengths[0]*voxel_size)

                # focal point
                p_0 = numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point

                # print numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point  # [ 1650.  5500.  5500.]
                # print numpy.array([Nx, Ny, Nz])  # [ 5500 11000 11000]
                # print self.configs.detector_focal_point  # (0.3, 0.5, 0.5)
                               

                # beam position : Assuming beam position = focal point for temporary
                p_b = copy.copy(p_0)


                # exposure time
                exposure_time = self.configs.detector_exposure_time

                start_time = self.configs.spatiocyte_start_time

                time = exposure_time * start_count
                end  = exposure_time * stop_count

                # data-time interval
		data_interval = self.configs.spatiocyte_interval
                delta_time = int(round(exposure_time/data_interval))

                # create frame data composed by frame element data
                count  = start_count
                count0 = int(round(start_time / exposure_time))

                # initialize Physical effects
                #length0 = len(self.configs.spatiocyte_data[0][1])
                #self.effects.set_states(t0, length0)

                while (time < end-1e-9) :
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


		    #if (numpy.amax(cell) > 0):
		    camera = self.detector_output(cell)
                     
		    # save data to numpy-binary file
		    image_file_name = os.path.join(self.configs.image_file_dir,
						self.configs.image_file_name_format % (count+500))
						#'image_%09.03f' % (self.configs.psf_intensity))
		    numpy.save(image_file_name, camera)

		    time  += exposure_time
		    count += 1



	def detector_output(self, cell) :

		# Detector Output
		voxel_radius = self.configs.spatiocyte_VoxelRadius
                voxel_size = (2.0*voxel_radius)/1e-9

		Nw_pixel = self.configs.detector_image_size[0]
		Nh_pixel = self.configs.detector_image_size[1]

		Np = int(self.configs.image_scaling*voxel_size)

                # image in nm-scale
		Nw_camera = Nw_pixel*Np
		Nh_camera = Nh_pixel*Np

		Nw_cell = len(cell)
		Nh_cell = len(cell[0])

		if (Nw_camera > Nw_cell) :
		    w_cam_from = int((Nw_camera - Nw_cell)/2.0)
		    w_cam_to   = w_cam_from + Nw_cell
                    w_cel_from = 0
                    w_cel_to   = Nw_cell
		else :
                    w_cam_from = 0
                    w_cam_to   = Nw_camera
                    w_cel_from = int((Nw_cell - Nw_camera)/2.0)
                    w_cel_to   = w_cel_from + Nw_camera

		if (Nh_camera > Nh_cell) :
                    h_cam_from = int((Nh_camera - Nh_cell)/2.0)
                    h_cam_to   = h_cam_from + Nh_cell
                    h_cel_from = 0
                    h_cel_to   = Nh_cell
                else :
                    h_cam_from = 0
                    h_cam_to   = int(Nh_camera)
                    h_cel_from = int((Nh_cell - Nh_camera)/2.0)
                    h_cel_to   = h_cel_from + Nh_camera


		# image in nm-scale
		plane = cell[w_cel_from:w_cel_to, h_cel_from:h_cel_to]

		# declear cell image in pixel-scale
                cell_pixel = numpy.zeros(shape=(Nw_cell/Np, Nh_cell/Np))

		# Signal (photon distribution on cell)
		for i in range(Nw_cell/Np) :
		    for j in range(Nh_cell/Np) :

			# get photons
			photons = numpy.sum(plane[i*Np:(i+1)*Np,j*Np:(j+1)*Np])
			cell_pixel[i][j] = photons

#			if (photons > 0) :
#
#			    # get crosstalk
#			    if (self.effects.detector_crosstalk_switch == True) :
#
#				width = self.effects.detector_crosstalk_width
#
#				n_i = numpy.random.normal(0, width, photons)
#				n_j = numpy.random.normal(0, width, photons)
#
#				#i_bins = int(numpy.amax(n_i) - numpy.amin(n_i))
#				#j_bins = int(numpy.amax(n_j) - numpy.amin(n_j))
#
#				smeared_photons, edge_i, edge_j = numpy.histogram2d(n_i, n_j, bins=(24, 24),
#                                                                                    range=[[-12,12],[-12,12]])
#
#				# smeared photon distributions
#				cell_pixel = self.overwrite_smeared(cell_pixel, smeared_photons, i, j)
#
#			    else :
#				cell_pixel[i][j] = photons

                # declear photon distribution for camera image
                #camera_pixel = numpy.zeros([Nw_pixel, Nh_pixel, 4])
                camera_pixel = numpy.zeros([Nw_pixel, Nh_pixel, 2])

		w_cam_from = int(w_cam_from/Np)
		w_cam_to   = int(w_cam_to/Np)
		h_cam_from = int(h_cam_from/Np)
		h_cam_to   = int(h_cam_to/Np)

		w_cel_from = int(w_cel_from/Np)
		w_cel_to   = int(w_cel_to/Np)
		h_cel_from = int(h_cel_from/Np)
		h_cel_to   = int(h_cel_to/Np)

		ddw = (w_cam_to - w_cam_from) - (w_cel_to - w_cel_from)
		ddh = (h_cam_to - h_cam_from) - (h_cel_to - h_cel_from)

		if   (ddw > 0) : w_cam_to = w_cam_to - ddw
		elif (ddw < 0) : w_cel_to = w_cel_to - ddw

                if   (ddh > 0) : h_cam_to = h_cam_to - ddh
                elif (ddh < 0) : h_cel_to = h_cel_to - ddh

		# place cell_pixel data to camera image
		camera_pixel[w_cam_from:w_cam_to, h_cam_from:h_cam_to, 0] = cell_pixel[w_cel_from:w_cel_to, h_cel_from:h_cel_to]

		print 'scaling [nm/pixel] :', Np
		print 'width  :', w_cam_from, w_cam_to
		print 'height :', h_cam_from, h_cam_to

		# set seed for random number
		numpy.random.seed()

		# CMOS (readout noise probability ditributions)
		if (self.configs.detector_type == "CMOS") :
#		    noise_data = numpy.loadtxt("catalog/detector/RNDist_F40.csv", delimiter=',')
                    noise_data = numpy.loadtxt(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                            'catalog/detector/RNDist_F40.csv'), delimiter=',')
		    Nr_cmos = noise_data[:,0]
		    p_noise = noise_data[:,1]
		    p_nsum  = p_noise.sum()


		# conversion : photon --> photoelectron --> ADC count
                for i in range(Nw_pixel) :
                    for j in range(Nh_pixel) :

			# pixel position
			pixel = (i, j)

			# Detector : Quantum Efficiency
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
			if (self.configs.detector_type == "CMOS") :

			    # get signal (poisson distributions)
			    signal = numpy.random.poisson(Exp, None)

			    #Noise = numpy.sqrt(self.signal_noise**2 + self.background_noise**2)
			    #signal = numpy.random.normal(Exp, Noise, None)
			    #if (signal < 0) : signal = 0

			    # get detector noise (photoelectrons)
			    noise  = numpy.random.choice(Nr_cmos, None, p=p_noise/p_nsum)
			    Nr = 1.3


			elif (self.configs.detector_type == "EMCCD") :

			    # get signal (photoelectrons)
			    if (Exp > 0) :

				# get EM gain
				M = self.configs.detector_emgain

				# set probability distributions
				s_min = M*int(Exp - 5.0*numpy.sqrt(Exp) - 10)
				s_max = M*int(Exp + 5.0*numpy.sqrt(Exp) + 10)

				if (s_min < 0) : s_min = 0

				s = numpy.array([k for k in range(s_min, s_max)])
		    		#p_signal = numpy.array(map(lambda x : self.prob_EMCCD(x, Exp), s))
				p_signal = self.prob_EMCCD(s, Exp)
				p_ssum = p_signal.sum()

				# get signal (photoelectrons)
				signal = numpy.random.choice(s, None, p=p_signal/p_ssum)

			    else :
				signal = 0

			    # get detector noise (photoelectrons)
			    Nr = self.configs.detector_readout_noise

			    if (Nr > 0) :
				noise = numpy.random.normal(0, Nr, None)
			    else : noise = 0


			elif (self.configs.detector_type == "CCD") :

			    # get signal (poisson distributions)
			    signal = numpy.random.poisson(Exp, None)

			    # get detector noise (photoelectrons)
			    Nr = self.configs.detector_readout_noise

			    if (Nr > 0) :
				noise = numpy.random.normal(0, Nr, None)
			    else : noise = 0

			# A/D converter : Expectation --> ADC counts
			EXP = self.get_ADC_value(pixel, Exp+Nr)

			# A/D converter : Photoelectrons --> ADC counts
			PE  = signal + noise
			ADC = self.get_ADC_value(pixel, PE)

			# set data in image array
			#camera_pixel[i][j] = [Photons, Exp, PE, ADC]
			camera_pixel[i][j] = [EXP, ADC]

                        
		return camera_pixel



