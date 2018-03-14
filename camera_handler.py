
import sys
import os
import copy
import tempfile
import math
import random
import numpy
import scipy

import parameter_configs
#from epifm_handler import VisualizerError, EPIFMConfigs, EPIFMVisualizer

from scipy.special import j0, i1, i1e


class VisualizerError(Exception):

    "Exception class for visualizer"

    def __init__(self, info):
        self.__info = info

    def __repr__(self):
        return self.__info

    def __str__(self):
        return self.__info



class CameraConfigs() :

    '''
    Camera Configuration : CCD/EMCCD/CMOS Cameras
    '''

    def __init__(self, user_configs_dict = None):

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



    def _set_data(self, key, val) :
    
        if val != None:
            setattr(self, key, val)



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
	print '\tGain = %.3f count/electron' % (self.ADConverter_gain)
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
	    offset = numpy.array([ADC0 for i in range(Nw_pixel*Nh_pixel)])

	if  (FPN_type == 'pixel') :
	    offset = numpy.rint(numpy.random.normal(ADC0, FPN_count, Nw_pixel*Nh_pixel))

	elif (FPN_type == 'column') :
	    column = numpy.random.normal(ADC0, FPN_count, Nh_pixel)
	    temporal = numpy.array([column for i in range(Nw_pixel)])

	    offset = numpy.rint(temporal.reshape(Nh_pixel*Nw_pixel))

	# set ADC gain
        gain = numpy.array(map(lambda x : (fullwell - 0.0)/(pow(2.0, bit) - x), offset))

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




class CameraVisualizer() :

	'''
	Camera Visualization class of e-cell simulator
	'''

	def __init__(self, configs=CameraConfigs()) :

		assert isinstance(configs, CameraConfigs)
		self.configs = configs

		"""
		Check and create the folders for image and output files.
		"""
		if not os.path.exists(self.configs.image_file_dir):
		    os.makedirs(self.configs.image_file_dir)
		#else:
		#    for file in os.listdir(self.configs.image_file_dir):
		#	os.remove(os.path.join(self.configs.image_file_dir, file))



        def output_frames(self, index=0, signal=0, background=0):

		Nw = self.configs.detector_image_size[0]
		Nh = self.configs.detector_image_size[1]

		photons = numpy.full([Nw, Nh], background)
		photons[int(0.2*Nw):int(0.8*Nw),int(0.2*Nh):int(0.8*Nh)] += numpy.full([int(0.6*Nw),int(0.6*Nh)], signal)

		camera = self.detector_output(photons)

#		# save data to numpy-binary file
#		image_file_name = os.path.join(self.configs.image_file_dir,
#				self.configs.image_file_name_format % (index))
#		numpy.save(image_file_name, camera)



	def prob_EMCCD(self, S, E) :

		# get EM gain
		M = self.configs.detector_emgain
		a = 1.00/M

		if (S > 0) :
		    prob = numpy.sqrt(a*E/S)*numpy.exp(-a*S-E+2*numpy.sqrt(a*E*S))*i1e(2*numpy.sqrt(a*E*S))
		else :
		    prob = numpy.exp(-E)

		return prob



	def detector_output(self, input_photons) :

		Nw_pixel = self.configs.detector_image_size[0]
		Nh_pixel = self.configs.detector_image_size[1]

                # declear photon distribution for camera image
                camera_pixel = numpy.zeros([Nw_pixel, Nh_pixel, 4])

		# set seed for random number
		numpy.random.seed()

		# CMOS (readout noise probability ditributions)
		if (self.configs.detector_type == "CMOS") :
		    noise_data = numpy.loadtxt("catalog/detector/RNDist_F40.csv", delimiter=',')
		    Nr = noise_data[:,0]
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
			Photons = input_photons[i][j]

			# get signal (expectation)
			Exp = QE*Photons

			# select Camera type
			if (self.configs.detector_type == "CMOS") :

			    # get signal (poisson distributions)
			    signal = numpy.random.poisson(Exp, None)

			    # get detector noise (photoelectrons)
			    noise  = numpy.random.choice(Nr, None, p=p_noise/p_nsum)


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


                        # get signal (photoelectrons)
			PE = signal + noise

			# A/D converter : Photoelectrons --> ADC counts
			ADC = self.get_ADC_value(pixel, PE)

			# set data in image array
			camera_pixel[i][j] = [Photons, Exp, PE, ADC]

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

	    if (ADC > ADC_max) :
		ADC = ADC_max

	    if (ADC < 0) :
		ADC = 0

	    return int(ADC)


