
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

from scipy.special import j0
from scipy.misc    import toimage


class LineScanConfocalConfigs(EPIFMConfigs) :

    '''
    Line-scanning Confocal configuration

	Line-like Gaussian Profile
	    +
	Line-scanning
	    +
	Slit
	    +
	Detector : EMCCD Camera
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



    def set_Slits(self, size = None) :

        print('--- Slit :')

        self._set_data('slit_size', size)

        print('\tSize = ', self.slit_size, 'm')



    def set_Illumination_path(self) :

        r = self.radial
        d = numpy.linspace(0, 20000, 20001)
        #d = self.depth

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

	print('Photon Flux Density (Max) :', numpy.amax(self.source_flux))



    def set_Detection_path(self) :

        wave_length = self.psf_wavelength*1e-9

	# Magnification
	Mag = self.image_magnification

	# set image scaling factor
        voxel_radius = self.spatiocyte_VoxelRadius

	# set zoom
	zoom = self.detector_zoom

	# set slit pixel length
	pixel_length = self.slit_size/(Mag*zoom)

	self.image_resolution = pixel_length
	self.image_scaling = pixel_length/(2.0*voxel_radius)

	print('Resolution :', self.image_resolution, 'm')
	print('Scaling :', self.image_scaling)

        # Detector PSF
        self.set_PSF_detector()



class LineScanConfocalVisualizer(EPIFMVisualizer) :

	'''
	Confocal Visualization class of e-cell simulator
	'''

	def __init__(self, configs=LineScanConfocalConfigs(), effects=PhysicalEffects()) :

		assert isinstance(configs, LineScanConfocalConfigs)
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
		source_psf = self.configs.source_flux[int(source_depth)][int(source_horizon)]
		#source_max = norm*self.configs.source_flux[0][0]

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



	def get_molecule_plane(self, cell, time, data, pid, p_b, p_0, offset) :

		voxel_size = (2.0*self.configs.spatiocyte_VoxelRadius)/1e-9

		# get beam position
		x_b, y_b, z_b = p_b

                # cutoff randius
		slit_radius = int(self.configs.image_scaling*voxel_size/2)
		cut_off = 3*slit_radius

		# particles coordinate, species and lattice IDs
                c_id, s_id, l_id = data

		sid_array = numpy.array(self.configs.spatiocyte_species_id)
		s_index = (numpy.abs(sid_array - int(s_id))).argmin()

		if self.configs.spatiocyte_observables[s_index] is True :

		    # Normalization
		    unit_time = 1.0
		    unit_area = (1e-9)**2
		    norm = (unit_area*unit_time)/(4.0*numpy.pi)

		    # particles coordinate in real(nm) scale
                    #pos = self.get_coordinate(c_id)
                    #p_i = numpy.array(pos)*voxel_size
                    p_i = self.get_coordinate(c_id)
		    x_i, y_i, z_i = p_i

		    if (abs(y_i - y_b) < cut_off) :

                        #print pid, s_id, p_i
                        # get signal matrix
                        signal = self.get_signal(time, pid, s_index, p_i, p_b, p_0, norm)

			z_from, y_from = offset
			z_size, y_size = cell.shape
			y_to = y_from + y_size
			z_to = z_from + z_size

			r = len(self.configs.radial)

                        # add signal matrix to image plane
			self.overwrite_signal(cell, signal, p_i, r, z_from, y_from, z_to, y_to)
                        #self.overwrite_signal(cell, signal, p_i)



        def overwrite_signal(self, cell, signal, p_i, r, z_from, y_from, z_to, y_to) :

            z_size = z_to - z_from
            y_size = y_to - y_from
            x_i, y_i, z_i = p_i
            zi_size, yi_size = signal.shape

            #z0_from = bounded(z_i - r - z_from, lower_bound=0, upper_bound=z_size)
            y0_from = bounded(y_i - r - y_from, lower_bound=0, upper_bound=y_size)
            #z0_to = bounded(z_i + r - z_to, lower_bound=0, upper_bound=z_size)
            y0_to = bounded(y_i + r - y_to, lower_bound=0, upper_bound=y_size)

            #zi_from = bounded(z_from - z_i + r, lower_bound=0, upper_bound=zi_size)
            yi_from = bounded(y_from - y_i + r, lower_bound=0, upper_bound=yi_size)
            #zi_to = bounded(z_from - z_i + r + z_size, lower_bound=0, upper_bound=zi_size)
            yi_to = bounded(y_from - y_i + r + y_size, lower_bound=0, upper_bound=yi_size)

	    # z-axis
	    Nz_cell  = len(cell)
	    Nz_signal = len(signal)
	    Nr = len(self.configs.radial)

            z_to   = z_i + Nr
            z_from = z_i - Nr

            if (z_to > Nz_cell) :

                dz_to = z_to - Nz_cell

                z0_to = int(Nz_cell)
                zi_to = int(Nz_signal - dz_to)

            else :

                dz_to = Nz_cell - (z_i + Nr)

                z0_to = int(Nz_cell - dz_to)
                zi_to = int(Nz_signal)

            if (z_from < 0) :

                dz_from = abs(z_from)

                z0_from = 0
                zi_from = int(dz_from)

            else :

                dz_from = z_from

                z0_from = int(dz_from)
                zi_from = 0

            ddz = (z0_to - z0_from) - (zi_to - zi_from)

            if (ddz > 0) : z0_to = z0_to - ddz
            if (ddz < 0) : zi_to = zi_to - ddz

            cell[z0_from:z0_to, y0_from:y0_to] += signal[zi_from:zi_to, yi_from:yi_to]



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

		# image dimenssion in pixel-scale
		Nw_pixel = self.configs.detector_image_size[0]
		Nh_pixel = self.configs.detector_image_size[1]

		# cells dimenssion in nm-scale
                Nz = int(self.configs.spatiocyte_lengths[2] * voxel_size)
                Ny = int(self.configs.spatiocyte_lengths[1] * voxel_size)
                Nx = int(self.configs.spatiocyte_lengths[0] * voxel_size)

		# pixel length : nm/pixel
		Np = int(self.configs.image_scaling*voxel_size)

		# cells dimenssion in pixel-scale
		Ny_pixel = Ny/Np
		Nz_pixel = Nz/Np

                # focal point
                p_0 = numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point

                # beam position : 
                beam_center = numpy.array(self.configs.detector_focal_point)

                # set boundary condition
#                if (self.configs.spatiocyte_bc_switch == True) :
#
#                    bc = numpy.zeros(shape=(Nz, Ny))
#                    bc = self.set_boundary_plane(bc, p_b, p_0)

		# exposure time on cell
		R = float(Ny_pixel) / float(Nw_pixel)

                exposure_time = self.configs.detector_exposure_time
                contact_time  = R * exposure_time
		non_contact_time = (exposure_time - contact_time)/2

                spatiocyte_start_time = self.configs.spatiocyte_start_time
                time = exposure_time * start_count
                end  = exposure_time * stop_count

                # data-time interval
		data_interval = self.configs.spatiocyte_interval

		# time/count
                delta_time = int(round(exposure_time / data_interval))

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

                    print('time : ', time, ' sec (', count, ')')

                    # define cell
                    cell = numpy.zeros(shape=(Nz, Ny))

                    count_start = (count - count0)*delta_time
                    count_end   = (count - count0 + 1)*delta_time

                    frame_data = self.configs.spatiocyte_data[count_start:count_end]

		    if (len(frame_data) > 0) :

		        # beam position : initial
		        p_b = numpy.array([Nx, Ny, Nz])*beam_center

		        # line-scanning sequences
		        scan_time = 0

		        while (scan_time < contact_time) :

			    # beam position : i-th frame
			    if (count%2 == 0) :
				# scan from left-to-right
				beam_center[1] = scan_time/contact_time
			    else :
				# scan from right-to-left
				beam_center[1] = 1 - scan_time/contact_time

			    p_b = numpy.array([Nx, Ny, Nz])*beam_center
			    x_b, y_b, z_b = p_b

                            # loop for frame data
			    i_time, i_data = frame_data[0]

			    diff = abs(i_time - (scan_time + time + non_contact_time))
			    data = i_data

                            for i, (i_time, i_data) in enumerate(frame_data) :
                                #print '\t', '%02d-th frame : ' % (i), i_time, ' sec'
			        i_diff = abs(i_time - (scan_time + time))

			        if (i_diff < diff) :
				    diff = i_diff
				    data = i_data

			    # overwrite the scanned region to cell
			    r_s = int(self.configs.image_scaling*voxel_size/2)

			    if (y_b-r_s < 0) : y_from = int(y_b)
			    else : y_from = int(y_b - r_s)

			    if (y_b+r_s >= Ny) : y_to = int(y_b)
			    else : y_to = int(y_b + r_s)

			    z_from, z_to = 0, Nz

			    offset = (z_from, y_from)
			    mask = numpy.zeros(shape=(z_to-z_from, y_to-y_from))

			    zz, yy = numpy.ogrid[z_from:z_to, y_from-int(y_b):y_to-int(y_b)]
			    rr_cut = yy**2 < r_s**2
			    mask[rr_cut] = 1

			    scan_cell = numpy.zeros_like(mask)

                            # loop for particles
                            for j, j_data in enumerate(data) :
                                self.get_molecule_plane(scan_cell, i_time, j_data, j, p_b, p_0, offset)

			    cell[z_from:z_to, y_from:y_to] += mask*scan_cell*(contact_time/Ny_pixel)

			    scan_time += contact_time/Ny_pixel


		    if (numpy.amax(cell) > 0) :

#			if (self.configs.spatiocyte_bc_switch == True) :
#			    camera = self.detector_output(cell, bc)
#			else : camera = self.detector_output(cell)

			camera = self.detector_output(cell)

			# save data to numpy-binary file
			image_file_name = os.path.join(self.configs.image_file_dir,
					self.configs.image_file_name_format % (count))
			numpy.save(image_file_name, camera)

			# save data to png-image file
			#camera[:,:,3].astype('uint%d' % (self.configs.ADConverter_bit))
			#toimage(camera[:,:,3], low=numpy.amin(camera[:,:,3]), high=numpy.amax(camera[:,:,3]), mode='I').save(image_file_name)

		    time  += exposure_time
		    count += 1



def bounded(value, lower_bound=None, upper_bound=None):

    if lower_bound is not None and value < lower_bound:
        return lower_bound

    if upper_bound is not None and value > upper_bound:
        return upper_bound

    return value
