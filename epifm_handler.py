
import sys
import os
import copy
import tempfile
import math
import operator
import random
#import h5py
import csv
import string
import ctypes
import multiprocessing

import scipy
import numpy

import parameter_configs
import parameter_effects

from effects_handler import PhysicalEffects

from scipy.special import j0, i1, i1e
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.misc    import toimage

class VisualizerError(Exception):

    "Exception class for visualizer"

    def __init__(self, info):
        self.__info = info

    def __repr__(self):
        return self.__info

    def __str__(self):
        return self.__info



class EPIFMConfigs() :

    '''
    EPIFM Configuration

	Wide-field Gaussian Profile
	    +
	Detector : EMCCD/CMOS
    '''

    def __init__(self, user_configs_dict = None):
        # 'source_switch': bool
        # 'source_type': str
        # 'source_wavelength': float
        # 'source_flux': float
        # 'source_radius': float
        # 'source_angle': float

        # 'excitation_switch': bool

        # 'fluorophore_type': str
        # 'psf_wavelength': float 
        # 'psf_intensity': 
        # 'psf_width': 
        # 'psf_cutoff': 
        # 'psf_file_name_format': 

        # 'dichroic_switch': bool
        # 'emission_switch': bool
	# 'image_magnification': float

        # 'detector_switch': bool
        # 'detector_type': str
        # 'detector_image_size': tupel[2]
        # 'detector_pixel_length': float
        # 'detector_focal_point': tuple[3]
        # 'detector_base_position': tuple[3]
        # 'detector_exposure_time': float
        # 'detector_qeff': float
        # 'detector_readout_noise': float
        # 'detector_dark_count': int
        # 'detector_emgain': int

	# 'ADConverter_bit': float
	# 'ADConverter_gain': float
	# 'ADConverter_offset': float
        # 'ADConverter_fullwell': float
        # 'ADConverter_fpn_type': 
        # 'ADConverter_fpn_count': float

	# 'image_file_dir': str
	# 'image_file_cleanup_dir': str

        # 'spatiocyte_start_time': float
        # 'spatiocyte_end_time': float
        # 'spatiocyte_interval': float
        # 'spatiocyte_data': list
        # 'spatiocyte_species_id': list  # [0, 1, ...]
        # 'spatiocyte_index': list  # ['A', 'B', ...]
	# 'spatiocyte_radius': list  # [1e-8, 1e-8, ...]
        # 'spatiocyte_lengths': list[3]
        # 'spatiocyte_VoxelRadius': float
        # 'spatiocyte_theNormalizedVoxelRadius': float

        # 'source_flux_density': numpy.array[20001][20001]


        # default setting
        configs_dict = parameter_configs.__dict__.copy()

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


    def _set_data(self, key, val):    
        if val != None:
            setattr(self, key, val)



    # def set_Particles(self, p_list = None) :
    #     print '--- Particles List :'
    #     self._set_data('particles_list', p_list)
    #     print p_list



    def set_LightSource(self,  source_type = None,
				wave_length = None,
                                flux = None,
                                radius = None,
                                angle  = None):

        self._set_data('source_switch', True)
        self._set_data('source_type', source_type)
        self._set_data('source_wavelength', wave_length)
        self._set_data('source_flux', flux)
        self._set_data('source_radius', radius)
        self._set_data('source_angle', angle)

        print '--- Light Source :', self.source_type
        print '\tWave Length = ', self.source_wavelength, 'nm'
        print '\tBeam Flux = ', self.source_flux, 'W (=joule/sec)'
        print '\t1/e2 Radius = ', self.source_radius, 'm'
        print '\tAngle = ', self.source_angle, 'degree'



    def set_ExcitationFilter(self, excitation = None) :
        print '--- Excitation Filter :'
        filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/excitation/') + excitation + '.csv'

        try:
            csvfile = open(filename)
            lines = csvfile.readlines()

            header = lines[0:5]
            data   = lines[6:]

            excitation_header = []
            excitation_filter = []

            for i in range(len(header)) :
                dummy  = header[i].split('\r\n')
                a_data = dummy[0].split(',')
                excitation_header.append(a_data)
                print '\t', a_data

            for i in range(len(data)) :
                dummy0 = data[i].split('\r\n')
                a_data = dummy0[0].split(',')
                excitation_filter.append(a_data)

        except Exception:
            print 'Error : ', filename, ' is NOT found'
            exit()

        ####
        self.excitation_eff = self.set_efficiency(excitation_filter)
        self._set_data('excitation_switch', True)



    def set_Fluorophore(self, fluorophore_type = None,
                              wave_length = None,
			      intensity = None,
                              width = None,
                              cutoff = None,
                              file_name_format = None ):

	if (fluorophore_type == 'Gaussian') :
            print '--- Fluorophore : Point Spreading Function [%s]' % (fluorophore_type)

            self._set_data('fluorophore_type', fluorophore_type)
            self._set_data('psf_wavelength', wave_length)
            self._set_data('psf_intensity', intensity)
            self._set_data('psf_width', width)
            self._set_data('psf_cutoff', cutoff)
            self._set_data('psf_file_name_format', file_name_format)

	    index = (numpy.abs(self.wave_length - self.psf_wavelength)).argmin()

	    self.fluoex_eff[index] = 100
	    self.fluoem_eff[index] = 100

            print '\tWave Length   = ', self.psf_wavelength, 'nm'
	    print '\tPeak Intensity = ', self.psf_intensity
            print '\tLateral Width = ', self.psf_width[0], 'nm'
            print '\tAxial Width = ', self.psf_width[1], 'nm'
	    print '\tLateral Cutoff = ', self.psf_cutoff[0], 'nm'
            print '\tAxial Cutoff = ', self.psf_cutoff[1], 'nm'

	elif (fluorophore_type == 'Point-like'):
            print '--- Fluorophore : Point Spreading Function [%s]' % (fluorophore_type)

            self._set_data('fluorophore_type', fluorophore_type)
            self._set_data('psf_wavelength', wave_length)
            self._set_data('psf_width', (10, 140))
            self._set_data('psf_file_name_format', file_name_format)

            index = (numpy.abs(self.wave_length - self.psf_wavelength)).argmin()

            self.fluoex_eff[index] = 100
            self.fluoem_eff[index] = 100

            print '\tWave Length   = ', self.psf_wavelength, 'nm'

	else :
	    print '--- Fluorophore :'
            filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/fluorophore/', fluorophore_type + '.csv')

            if not os.path.exists(filename):
                print 'Error : ', filename, ' is NOT found'
                exit()
                
            with open(filename) as csvfile:
                lines = [string.rstrip(_) for _ in csvfile.readlines()]

                header = lines[0:5]
                data   = lines[5:]

                fluorophore_header     = [_.split(',') for _ in header]
                for _ in fluorophore_header: print "\t", _

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
            self._set_data('psf_file_name_format', file_name_format)

            self.fluoex_eff[index_ex] = 100
            self.fluoem_eff[index_em] = 100

            print '\tExcitation : Wave Length = ', self.wave_length[index_ex], 'nm'
            print '\tEmission   : Wave Length = ', self.psf_wavelength, 'nm'
                        

	# Normalization
	norm = sum(self.fluoex_eff)
	self.fluoex_norm = numpy.array(self.fluoex_eff)/norm

	norm = sum(self.fluoem_eff)
	self.fluoem_norm = numpy.array(self.fluoem_eff)/norm



    def set_DichroicMirror(self, dm = None) :
        print '--- Dichroic Mirror :'
	filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/dichroic/') + dm + '.csv'

	try:
	    csvfile = open(filename)
	    lines = csvfile.readlines()

	    header = lines[0:5]
	    data   = lines[6:]

	    dichroic_header = []
	    dichroic_mirror = []

	    for i in range(len(header)) :
		dummy  = header[i].split('\r\n')
		a_data = dummy[0].split(',')
		dichroic_header.append(a_data)
		print '\t', a_data

	    for i in range(len(data)) :
		dummy0 = data[i].split('\r\n')
		a_data = dummy0[0].split(',')
		dichroic_mirror.append(a_data)

        except Exception:
            print 'Error : ', filename, ' is NOT found'
	    exit()

        self.dichroic_eff = self.set_efficiency(dichroic_mirror)
	self._set_data('dichroic_switch', True)



    def set_EmissionFilter(self, emission = None) :
        print '--- Emission Filter :'
        filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'catalog/emission/') + emission + '.csv'

        try:
            csvfile = open(filename)
            lines = csvfile.readlines()

            header = lines[0:5]
            data   = lines[6:]

	    emission_header = []
	    emission_filter = []

            for i in range(len(header)) :
                dummy  = header[i].split('\r\n')
                a_data = dummy[0].split(',')
                emission_header.append(a_data)
		print '\t', a_data

            for i in range(len(data)) :
                dummy0 = data[i].split('\r\n')
                a_data = dummy0[0].split(',')
                emission_filter.append(a_data)

        except Exception:
            print 'Error : ', filename, ' is NOT found'
            exit()

        self.emission_eff = self.set_efficiency(emission_filter)
        self._set_data('emission_switch', True)


    def set_Magnification(self, Mag = None) :
	self._set_data('image_magnification', Mag)
	print '--- Magnification : x', self.image_magnification


    def set_Detector(self, detector = None,
		   image_size = None,
                   pixel_length = None,
                   focal_point = None,
                   base_position = None,
                   exposure_time = None,
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
        self._set_data('detector_base_position', base_position)
        self._set_data('detector_exposure_time', exposure_time)
        self._set_data('detector_qeff', QE)
        self._set_data('detector_readout_noise', readout_noise)
        self._set_data('detector_dark_count', dark_count)
        self._set_data('detector_emgain', emgain)

	print '--- Detector : ', self.detector_type
        print '\tImage Size  = ', self.detector_image_size[0], 'x', self.detector_image_size[1]
        print '\tPixel Size  = ', self.detector_pixel_length, 'm/pixel'
        print '\tFocal Point = ', self.detector_focal_point
        print '\tPosition    = ', self.detector_base_position
        print '\tExposure Time = ', self.detector_exposure_time, 'sec'
        print '\tQuantum Efficiency = ', 100*self.detector_qeff, '%'
	print '\tReadout Noise = ', self.detector_readout_noise, 'electron'
        print '\tDark Count = ', self.detector_dark_count, 'electron/sec'
        print '\tEM gain = ', 'x', self.detector_emgain
                    


    def set_ADConverter(self, bit = None,
			gain = None,
			offset = None,
			fullwell = None,
			fpn_type = None,
			fpn_count = None) :

	self._set_data('ADConverter_bit', bit)
	self._set_data('ADConverter_gain', (fullwell - 0.0)/(pow(2.0, bit) - offset))
	self._set_data('ADConverter_offset', offset)
        self._set_data('ADConverter_fullwell', fullwell)
        self._set_data('ADConverter_fpn_type', fpn_type)
        self._set_data('ADConverter_fpn_count', fpn_count)

        print '--- A/D Converter : %d-bit' % (self.ADConverter_bit)
	print '\tGain = %.3f electron/count' % (self.ADConverter_gain)
	print '\tOffset = ', self.ADConverter_offset, 'count'
        print '\tFullwell = ', self.ADConverter_fullwell, 'electron'
        print '\t%s-Fixed Pattern Noise :' % (self.ADConverter_fpn_type), self.ADConverter_fpn_count, 'count'


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


	if (FPN_type == None) :
            # offset = numpy.empty(Nw_pixel*Nh_pixel)
            # offset.fill(ADC0)
	    offset = numpy.array([ADC0 for i in range(Nw_pixel*Nh_pixel)])
            
	elif  (FPN_type == 'pixel') :
	    offset = numpy.rint(numpy.random.normal(ADC0, FPN_count, Nw_pixel*Nh_pixel))

	elif (FPN_type == 'column') :
	    column = numpy.random.normal(ADC0, FPN_count, Nh_pixel)
	    temporal = numpy.array([column for i in range(Nw_pixel)])

	    offset = numpy.rint(temporal.reshape(Nh_pixel*Nw_pixel))

	# set ADC gain
#        gain = numpy.array(map(lambda x : (fullwell - 0.0)/(pow(2.0, bit) - x), offset))
        gain = (fullwell - 0.0)/(pow(2.0, bit) - offset)

	# reshape
        self.ADConverter_offset = offset.reshape([Nw_pixel, Nh_pixel])
        self.ADConverter_gain = gain.reshape([Nw_pixel, Nh_pixel])



    def set_OutputData(self, image_file_dir = None,
			image_file_cleanup_dir=False) :

	if image_file_dir is None :

	    image_file_dir = tempfile.mkdtemp(dir=os.getcwd())
	    image_file_cleanup_dir = True
            
	self._set_data('image_file_dir', image_file_dir)
	self._set_data('image_file_cleanup_dir', image_file_cleanup_dir)



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
		    particles.append((c_id, s_id, l_id))
                    
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
        #self._set_data('spatiocyte_theStartCoord', lattice[0][4])
        #self._set_data('spatiocyte_theRowSize',    lattice[0][6])
        #self._set_data('spatiocyte_theLayerSize',  lattice[0][5])
        #self._set_data('spatiocyte_theColSize',    lattice[0][7])

       
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


    def set_efficiency(self, array, index=1) :

# 	if (len(array[0]) < 3) : index = 1
#
#         N = len(self.wave_length)
#
# #        efficiency = numpy.array([0.0 for i in range(N)])
#         efficiency = [0.0 for i in range(N)]
#
#         for i in range(N) :
#             wl = self.wave_length[i]
#
#             for j in range(len(array)) :
#
#                 length = float(array[j][0])
#                 eff = float(array[j][index])
#
#                 if (length/wl == 1) :
#                     efficiency[i] = eff
#                    
#         return efficiency

        array = numpy.array(array, dtype = 'float')
        array = array[array[:, 0] % 1 == 0,:] 

        efficiency = numpy.zeros(len(self.wave_length))
        idx1 = numpy.in1d(numpy.array(self.wave_length), array[:, 0])
        idx2 = numpy.in1d(numpy.array(array[:, 0]), self.wave_length)
        
        efficiency[idx1] = array[idx2, 1]
                
        return efficiency.tolist()



    def set_Optical_path(self) :
	# (1) Illumination path : Light source --> Sample
	self.set_Illumination_path()
	# (2) Detection path : Sample --> Detector
	self.set_Detection_path()



    def set_Illumination_path(self) :
        #r = self.radial
        #d = self.depth
	r = numpy.linspace(0, 20000, 20001)
	d = numpy.linspace(0, 20000, 20001)

        # (plank const) * (speed of light) [joules meter]
        hc = 2.00e-25

	# Illumination
        w_0 = self.source_radius

	# flux [joules/sec]
        P_0 = self.source_flux

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
        self.source_flux_density = 2*N_0/(numpy.pi*w_z[:, numpy.newaxis]**2)*numpy.exp(-2*(r*1e-9/w_z[:, numpy.newaxis])**2)
#        self.source_flux_density = numpy.array(map(lambda x : 2*N_0/(numpy.pi*x**2)*numpy.exp(-2*(r*1e-9/x)**2), w_z))

	print 'Photon Flux Density (Max) :', numpy.amax(self.source_flux_density)


    def set_Detection_path(self) :
        wave_length = self.psf_wavelength*1e-9

	# Magnification
	Mag = self.image_magnification

	# set image scaling factor
        voxel_radius = self.spatiocyte_VoxelRadius

	# set camera's pixel length
	pixel_length = self.detector_pixel_length/Mag

	self.image_resolution = pixel_length
	self.image_scaling = pixel_length/(2.0*voxel_radius)

	print 'Resolution :', self.image_resolution, 'm'
	print 'Scaling :', self.image_scaling

        # Detector PSF
        self.set_PSF_detector()



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
	psf_fl = None

	if (self.fluorophore_type == 'Gaussian' or
	    self.fluorophore_type == 'Point-like' ) :

            I0 = self.psf_intensity
            Ir = sum(map(lambda x : x*numpy.exp(-0.5*(r/self.psf_width[0])**2), norm))
            Iz = sum(map(lambda x : x*numpy.exp(-0.5*(z/self.psf_width[1])**2), norm))

	    psf_fl = numpy.array(map(lambda x : I0*Ir*x, Iz))

	else :
	    # make the norm and wave_length array shorter
#	    psf_fl = 0
#
#	    for i in range(len(norm)) :
#
#		if norm[i] is True :
#	            psf_fl += self.get_PSF_fluorophore(r, z, wave_length[i])
#
#	    psf_fl = psf_fl/sum(norm)
	    psf_fl = numpy.sum(I)*self.get_PSF_fluorophore(r, z, wave_length)

	self.fluorophore_psf = psf_fl



    def get_PSF_fluorophore(self, r, z, wave_length) :
	# set Magnification of optical system
	M = self.image_magnification

	# set Numerical Appature
	NA = 1.4#self.objective_NA

	# set alpha and gamma consts
	k = 2.0*numpy.pi/wave_length
	alpha = k*NA
	gamma = k*(NA/2)**2

	# set rho parameters
	N = 80
        drho = 1.0/N
        rho = numpy.array([(i+1)*drho for i in range(N)])

        J0 = numpy.array(map(lambda x : j0(x*alpha*rho), r))
        Y  = numpy.array(map(lambda x : 2*numpy.exp(-2*1.j*x*gamma*rho**2)*rho*drho, z))
#	J0 = numpy.array(map(lambda y : map(lambda x : j0(x*y*rho), r), alpha))
#	Y  = numpy.array(map(lambda y : map(lambda x : 2*numpy.exp(-2*1.j*x*y*rho**2)*rho*drho, z), gamma))

        I  = numpy.array(map(lambda x : x*J0, Y))
        I_sum = I.sum(axis=2)

        psf = numpy.array(map(lambda x : abs(x)**2, I_sum))
	Norm = numpy.amax(psf)

#	for i in range(len(wave_length)) :
#
#	    I  = numpy.array(map(lambda x : x*J0[i], Y[i]))
#	    I_sum = I.sum(axis=2)
#	    I_abs = map(lambda x : abs(x)**2, I_sum)
#
#	    if (i > 0) : psf += I_abs
#	    else : psf = I_abs

	return psf/Norm


class EPIFMVisualizer() :

	'''
	EPIFM Visualization class of e-cell simulator
	'''

	def __init__(self, configs=EPIFMConfigs(), effects=PhysicalEffects()) :
           
                assert isinstance(configs, EPIFMConfigs)
		self.configs = configs

                assert isinstance(effects, PhysicalEffects)
                self.effects = effects
                
		"""
		Check and create the folders for image and output files.
		"""
		if not os.path.exists(self.configs.image_file_dir):
		    os.makedirs(self.configs.image_file_dir)

                """
                set Optical path from light source to detector
                """
		self.configs.set_Optical_path()


	def get_coordinate(self, aCoord) :
                point_y = aCoord[1]/1e-9
                point_z = aCoord[2]/1e-9
                point_x = aCoord[0]/1e-9

                return point_x, point_y, point_z



        def polar2cartesian_coordinates(self, r, t, x, y) :
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



        def polar2cartesian(self, grid, coordinates, shape) :
            r = shape[0] - 1
            psf_cart = numpy.empty([2 * r + 1, 2 * r + 1])
            psf_cart[r:, r:] = map_coordinates(grid, coordinates, order=0).reshape(shape)
            psf_cart[r:, :r] = psf_cart[r:, :r:-1]
            psf_cart[:r, :] = psf_cart[:r:-1, :]

            return psf_cart


	def get_intensity(self, time, pid, source_psf, source_max) :
		delta = self.configs.detector_exposure_time

                # linear conversion of intensity
                Ratio = self.effects.conversion_ratio
                intensity = Ratio*source_psf

		# Photobleaching process : Exponential decay
		if (self.effects.bleaching_switch == True) :
		    zeta  = self.effects.bleaching_rate
		    state = source_psf/source_max

		    self.effects.bleaching_state[pid] -= zeta*state*self.effects.bleaching_state[pid]*delta

                    intensity = self.effects.bleaching_state[pid]*intensity


                # Slow-Blinking process : Power-law probability distribution
                if (self.effects.blinking_switch == True) :

		    state  = self.effects.blinking_state[pid]
		    period = self.effects.blinking_period[pid]

		    value = scipy.random.uniform(0, 1)
		    prob = self.effects.get_Prob_blinking(state, period)

		    if (value > prob) :
		        self.effects.blinking_state[pid]  = int(bool(self.effects.blinking_state[pid]-1))
		        self.effects.blinking_period[pid] = 0
		    else :
		        self.effects.blinking_period[pid] += delta

                    intensity = self.effects.blinking_state[pid]*intensity


		return intensity



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
                """ Distance between beam and fluorefore (depth) """
		d_s = abs(x_i - x_b)

                if (d_s < 20000) :
		    source_depth = d_s
                else :
		    source_depth = 19999

		# beam position
                """ Distance between beam position and fluorefore (plane) """
		rr = numpy.sqrt((y_i-y_b)**2 + (z_i-z_b)**2)

		if (rr < 20000) :
		    source_radius = rr
		else :
		    source_radius = 19999

		# get illumination PSF
		source_psf = self.configs.source_flux_density[int(source_depth)][int(source_radius)]
		#source_max = norm*self.configs.source_flux_density[0][0]

                # signal conversion : Output Intensity = Physics * PSF (Beam)
		#Intensity = self.get_intensity(time, pid, source_psf, source_max)
		Ratio = self.effects.conversion_ratio

		# fluorophore axial position
		d_f = abs(x_i - x_b)

		if (d_f < len(d)) :
		    fluo_depth = d_f
		else :
		    fluo_depth = d[-1]

#		# coordinate transformation : polar --> cartisian
#		theta = numpy.linspace(0, 180, 181)
#
#                z = numpy.linspace(0, +r[-1], len(r))
#                y = numpy.linspace(-r[-1], +r[-1], 2*len(r)-1)
#
#		psf_t = numpy.array(map(lambda x : 1.00, theta))
#		psf_r = self.configs.fluorophore_psf[int(fluo_depth)]
#
#		psf_polar = numpy.array(map(lambda x : psf_t*x, psf_r))
#
#                # get fluorophore PSF
#		fluo_psf  = numpy.array(self.polar2cartesian(r, theta, psf_polar, z, y))

		# get fluorophore PSF
		fluo_psf = self.fluo_psf[int(fluo_depth)]

                # signal conversion : Output PSF = PSF(source) * Ratio * PSF(Fluorophore)
		signal = norm * source_psf * Ratio * fluo_psf

                return signal


	def overwrite_signal(self, cell, signal, p_i) :

                # particle position
                x_i, y_i, z_i = p_i

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
                if (ddz < 0) : zi_to = zi_to + ddz

                # y-axis
                Ny_cell  = cell.size/Nz_cell
                Ny_signal = signal.size/Nz_signal

		y_to   = y_i + Nr
		y_from = y_i - Nr

                if (y_to > Ny_cell) :
                    dy_to = y_to - Ny_cell
                    y0_to = int(Ny_cell)
                    yi_to = int(Ny_signal - dy_to)
                else :
                    dy_to = Ny_cell - (y_i + Nr)
                    y0_to = int(Ny_cell - dy_to)
                    yi_to = int(Ny_signal)

                if (y_from < 0) :
                    dy_from = abs(y_from)
                    y0_from = 0
                    yi_from = int(dy_from)
                else :
                    dy_from = y_from
                    y0_from = int(dy_from)
                    yi_from = 0

		ddy = (y0_to - y0_from) - (yi_to - yi_from)

		if (ddy > 0) : y0_to = y0_to - ddy
		if (ddy < 0) : yi_to = yi_to + ddy

		# add to cellular plane
                cell[z0_from:z0_to, y0_from:y0_to] += signal[zi_from:zi_to, yi_from:yi_to]



	def get_molecule_plane(self, cell, time, data, pid, p_b, p_0) :

		voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9
                
		# particles coordinate, species and lattice IDs
                c_id, s_id, l_id = data
                
		sid_array = numpy.array(self.configs.spatiocyte_species_id)
		s_index = (numpy.abs(sid_array - int(s_id))).argmin()

		if self.configs.spatiocyte_observables[s_index] is True :

		    # Normalization
                    """ Input: #photon / nm^2 * sec """
		    unit_time = self.configs.spatiocyte_interval
		    unit_area = (1e-9)**2
		    norm = unit_area*unit_time/(4.0*numpy.pi)

		    # particles coordinate in real(nm) scale
                    #pos = self.get_coordinate(c_id)
                    #p_i = numpy.array(pos)*voxel_size
                    p_i = self.get_coordinate(c_id)

                    # get signal matrix
                    signal = self.get_signal(time, pid, s_index, p_i, p_b, p_0, norm)

                    # add signal matrix to image plane
                    self.overwrite_signal(cell, signal, p_i)



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
#            if True:
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


		    if (numpy.amax(cell) > 0):
			camera = self.detector_output(cell)
                        
			# save data to numpy-binary file
			image_file_name = os.path.join(self.configs.image_file_dir,
                                                       self.configs.image_file_name_format % (count))
			numpy.save(image_file_name, camera)

		    time  += exposure_time
		    count += 1



	def overwrite_smeared(self, cell_pixel, photon_dist, i, j) :
		# i-th pixel
		Ni_pixel = len(cell_pixel)
		Ni_pe    = len(photon_dist)

                i_to   = i + Ni_pe/2
                i_from = i - Ni_pe/2

                if (i_to > Ni_pixel) :
                    di_to = i_to - Ni_pixel
                    i0_to = int(Ni_pixel)
                    i1_to = int(Ni_pe - di_to)
                else :
                    di_to = Ni_pixel - (i + Ni_pe/2)
                    i0_to = int(Ni_pixel - di_to)
                    i1_to = int(Ni_pe)

                if (i_from < 0) :
                    di_from = abs(i_from)
                    i0_from = 0
                    i1_from = int(di_from)
                else :
                    di_from = i_from
                    i0_from = int(di_from)
                    i1_from = 0

                ddi = (i0_to - i0_from) - (i1_to - i1_from)

                if (ddi > 0) : i0_to = i0_to - ddi
                if (ddi < 0) : i1_to = i1_to - ddi

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

                if (j_from < 0) :
                    dj_from = abs(j_from)
                    j0_from = 0
                    j1_from = int(dj_from)
                else :
                    dj_from = j_from
                    j0_from = int(dj_from)
                    j1_from = 0

                ddj = (j0_to - j0_from) - (j1_to - j1_from)

                if (ddj > 0) : j0_to = j0_to - ddj
                if (ddj < 0) : j1_to = j1_to - ddj

		# add to cellular plane
                cell_pixel[i0_from:i0_to, j0_from:j0_to] += photon_dist[i1_from:i1_to, j1_from:j1_to]

		return cell_pixel



	def prob_EMCCD(self, S, E) :

		# get EM gain
		M = self.configs.detector_emgain
		a = 1.00/M

		if (S > 0):
		    prob = numpy.sqrt(a*E/S)*numpy.exp(-a*S-E+2*numpy.sqrt(a*E*S))*i1e(2*numpy.sqrt(a*E*S))
		else:
		    prob = numpy.exp(-E)

		return prob



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

			if (photons > 0) :

			    # get crosstalk
			    if (self.effects.detector_crosstalk_switch == True) :

				width = self.effects.detector_crosstalk_width

				n_i = numpy.random.normal(0, width, photons)
				n_j = numpy.random.normal(0, width, photons)

				#i_bins = int(numpy.amax(n_i) - numpy.amin(n_i))
				#j_bins = int(numpy.amax(n_j) - numpy.amin(n_j))

				smeared_photons, edge_i, edge_j = numpy.histogram2d(n_i, n_j, bins=(24, 24),
                                                                                    range=[[-12,12],[-12,12]])

				# smeared photon distributions
				cell_pixel = self.overwrite_smeared(cell_pixel, smeared_photons, i, j)

			    else :
				cell_pixel[i][j] = photons


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
                        if (self.effects.background_switch == True) :
                            Photons_bg = self.effects.background_mean
                            Photons += Photons_bg

			# get signal (expectation)
			Exp = QE*Photons

			# select Camera type
			if (self.configs.detector_type == "CMOS") :

			    # get signal (poisson distributions)
			    signal = numpy.random.poisson(Exp, None)

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
		    		p_signal = numpy.array(map(lambda x : self.prob_EMCCD(x, Exp), s))
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



	def get_ADC_value(self, pixel, photoelectron) :
	    # pixel position
	    i, j = pixel

	    # check non-linearity
	    fullwell = self.configs.ADConverter_fullwell

	    if (photoelectron > fullwell) :
		photoelectron = fullwell

            # convert photoelectron to ADC counts
            k = self.configs.ADConverter_gain[i][j]
            ADC0 = self.configs.ADConverter_offset[i][j]
            ADC_max = 2**self.configs.ADConverter_bit - 1

            ADC = photoelectron/k + ADC0

	    if (ADC > ADC_max):	ADC = ADC_max
	    if (ADC < 0): ADC = 0

	    return int(ADC)



        def use_multiprocess(self) :
            envname = 'ECELL_MICROSCOPE_SINGLE_PROCESS'
            return not (envname in os.environ and os.environ[envname])



        def set_fluo_psf(self) :

            depths = set()

            voxel_size = 2.0*self.configs.spatiocyte_VoxelRadius/1e-9

            start = self.configs.spatiocyte_start_time
            end = self.configs.spatiocyte_end_time

            exposure_time = self.configs.detector_exposure_time
            num_timesteps = int(math.ceil((end - start) / exposure_time))

            t0 = self.configs.spatiocyte_data[0][0]

	    if (len(self.configs.spatiocyte_data) > 1) :
                t1 = self.configs.spatiocyte_data[1][0]
	    else :
		t1 = t0 + exposure_time

            delta_data = t1 - t0
            delta_time = int(round(exposure_time / delta_data))

            count0 = int(round(start / exposure_time))

	    for count in range(count0, count0+num_timesteps) :

                Nz = int(self.configs.spatiocyte_lengths[2] * voxel_size)
                Ny = int(self.configs.spatiocyte_lengths[1] * voxel_size)
                Nx = int(self.configs.spatiocyte_lengths[0] * voxel_size)

                # focal point
                p_0 = numpy.array([Nx, Ny, Nz])*self.configs.detector_focal_point

                count_start = (count - count0) * delta_time
                count_end = (count - count0 + 1) * delta_time

                frame_data = self.configs.spatiocyte_data[count_start:count_end]

                for _, data in frame_data :
                    for data_j in data :

                        c_id = data_j[0]
                        p_i = self.get_coordinate(c_id)
                        #p_i = p_0

                        # fluorophore axial position
                        d = self.configs.depth
                        d_f = abs(p_i[0] - p_0[0])

                        if d_f < len(d):
                            fluo_depth = d_f
                        else:
                            fluo_depth = d[-1]

                        depths.add(int(fluo_depth))

            depths = list(depths)

            if self.use_multiprocess():

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



        def get_fluo_psf(self, depths) :
            r = self.configs.radial
            theta = numpy.linspace(0, 90, 91)

            z = numpy.linspace(0, +r[-1], len(r))
            y = numpy.linspace(0, +r[-1], len(r))

            coordinates = self.polar2cartesian_coordinates(r, theta, z, y)
            psf_t = numpy.ones_like(theta)
            result = {}

            for depth in depths:
                psf_r = self.configs.fluorophore_psf[depth]
                psf_polar = numpy.array(map(lambda x: psf_t * x, psf_r))
                result[depth] = self.polar2cartesian(psf_polar, coordinates, (len(r), len(r)))

            return result



        def get_fluo_psf_process(self, depths, pipe) :
            pipe.send(self.get_fluo_psf(depths))
