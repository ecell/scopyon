
import sys
import os
import copy
import tempfile
import time
import math
import random

import scipy
import numpy

import parameter_configs
#from effects_handler import PhysicalEffects
#from fcs_handler import VisualizerError, FCSConfigs, FCSVisualizer

from scipy.special import j0, gamma
from scipy.misc    import toimage


class VisualizerError(Exception):

    "Exception class for visualizer"

    def __init__(self, info):
        self.__info = info

    def __repr__(self):
        return self.__info

    def __str__(self):
        return self.__info



class PMTConfigs() :

    '''
    PMT configuration : Photomultipliers Tube (PMT)
    '''

    def __init__(self, user_configs_dict = None):

        # default setting
        configs_dict = parameter_configs.__dict__.copy()

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



    def _set_data(self, key, val) :
    
        if val != None:
            setattr(self, key, val)



    def set_Detector(self, detector = None,
		   mode = None,
                   bandwidth = None,
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
        self._set_data('detector_bandwidth', bandwidth)
        self._set_data('detector_qeff', QE)
        self._set_data('detector_readout_noise', readout_noise)
        self._set_data('detector_dark_count', dark_count)
        self._set_data('detector_gain', gain)
        self._set_data('detector_dyn_stages', dyn_stages)
        self._set_data('detector_pair_pulses', pair_pulses)

	print('--- Detector : ', self.detector_type, ' (', self.detector_mode, 'mode )')
        print('\tBandwidth = ', self.detector_exposure_time, 'Hz')
        print('\tQuantum Efficiency = ', 100*self.detector_qeff, '%')
	print('\tReadout Noise = ', self.detector_readout_noise, 'electron/sec')
        print('\tDark Count = ', self.detector_dark_count, 'electron/sec')
        print('\tGain = ', 'x', self.detector_gain)
        print('\tDynode = ', self.detector_dyn_stages, 'stages')
        print('\tPair-pulses = ', self.detector_pair_pulses, 'sec')



    def set_OutputData(self, image_file_dir = None,
			image_file_cleanup_dir=False) :

	if image_file_dir is None :

	    image_file_dir = tempfile.mkdtemp(dir=os.getcwd())
	    image_file_cleanup_dir = True
            
	self._set_data('image_file_dir', image_file_dir)
	self._set_data('image_file_cleanup_dir', image_file_cleanup_dir)




class PMTVisualizer() :

	'''
	PMT Visualization class
	'''

	def __init__(self, configs=PMTConfigs()) :

		assert isinstance(configs, PMTConfigs)
		self.configs = configs

		"""
		Check and create the folder for image file.
		"""
		if not os.path.exists(self.configs.image_file_dir):
		    os.makedirs(self.configs.image_file_dir)
		#else:
		#    for file in os.listdir(self.configs.movie_image_file_dir):
		#	os.remove(os.path.join(self.configs.movie_image_file_dir, file))


        def output_frames(self, index=0, signal=0, background=0, duration=None):

		self.configs.image_file_name_format = "./output_%07d.dat"

                image_file_name = os.path.join(self.configs.image_file_dir,
				self.configs.image_file_name_format % (index))

		with open(image_file_name, 'w') as output :
		    output.write('#time\tphotons\t\n')
		    output.write('\n')

		# PMT bandwidth
		B = self.configs.detector_bandwidth
		T = 1/(2*B)

		# initial time
		time = 0

                while (time < duration) :

		    # get the number of photons detected in the PMT
		    photons = self.detector_output(signal + background)

		    with open(image_file_name, 'a') as output :
			line  = str(time)	+ '\t'
			line += str(photons)	+ '\t\n'
			output.write(line)

		    time += T



#        def output_frames(self, index=0, signal=0, background=0, image_size=(100,100)):
#
#		Nw = image_size[0]
#		Nh = image_size[1]
#
#		flux = numpy.full([Nw, Nh], background)
#		flux[int(0.2*Nw):int(0.8*Nw),int(0.2*Nh):int(0.8*Nh)] += numpy.full([int(0.6*Nw),int(0.6*Nh)], signal)
#
#                # declear photon distribution for camera image
#                camera = numpy.zeros([Nw, Nh, 4])
#
#		for i in range(Nw) :
#		    for j in range(Nh) :
#
#			camera[i][j] = self.detector_output(flux[i][j])
#
#		# save data to numpy-binary file
#		image_file_name = os.path.join(self.configs.image_file_dir,
#				self.configs.image_file_name_format % (index))
#		numpy.save(image_file_name, camera)



	def prob_analog(self, y, alpha) :

		# get average gain
		A  = self.configs.detector_gain

		# get dynode stages
		nu = self.configs.detector_dyn_stages

		B = 0.5*(A - 1)/(A**(1.0/nu) - 1)
		c = numpy.exp(alpha*(numpy.exp(-A/B) - 1))

		m_y = alpha*A
		m_x = m_y/(1 - c)

		s2_y = alpha*(A**2 + 2*A*B)
		s2_x = s2_y/(1 - c) - c*m_x**2

		# Rayleigh approximation
		#s2 = (2.0/numpy.pi)*m_x**2
		#prob = y/s2*numpy.exp(-0.5*y**2/s2)

		if (y < 100*A) :
		    # Gamma approximation
		    k_1 = m_x
		    k_2 = (m_y**2 + s2_y)/(1 - c)

		    a = 1/(k_1*(k_2/k_1**2 - 1))
		    b = a*k_1

		    prob = a/gamma(b)*(a*y)**(b-1)*numpy.exp(-a*y)

		else :
		    # Truncated Gaussian approximation
		    Q = 0
		    beta0 = m_x/numpy.sqrt(s2_x)
		    beta  = beta0
		    delta = 0.1*beta0

		    while (beta < 11*beta0) :
			Q += numpy.exp(-0.5*beta**2)/numpy.sqrt(2*numpy.pi)*delta
			beta += delta

		    prob = numpy.exp(-0.5*(y - m_x)**2/s2_x)/(numpy.sqrt(2*numpy.pi*s2_x)*(1 - Q))
		    #print alpha, Q, prob

		return prob



	def detector_output(self, photon_flux) :

		# set seed for random number
		numpy.random.seed()

		# observational time
		T = self.configs.detector_exposure_time

		# Detector : Quantum Efficiency
		QE = self.configs.detector_qeff

                # get signal (photon flux)
		Flux = photon_flux

		if (self.configs.detector_mode == "Photon-counting") :
		    # pair-pulses time resolution (sec)
		    t_pp = self.configs.detector_pair_pulses
		    Flux = Flux/(1 + Flux*t_pp)

		Photons = Flux*T

		# get signal (expectation)
		Exp = QE*Photons

		# get dark count
		D = self.configs.detector_dark_count
		Exp += D*T

		# select Camera type
		if (self.configs.detector_mode == "Photon-counting") :

		    # get signal (poisson distributions)
		    signal = numpy.random.poisson(Exp, None)


		elif (self.configs.detector_mode == "Analog") :

		    # get signal (photoelectrons)
		    if (Exp > 0) :

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

			# probability density function
		    	p_signal = numpy.array(map(lambda y : self.prob_analog(y, Exp), s))
			p_ssum = p_signal.sum()

			# get signal (photoelectrons)
			signal = numpy.random.choice(s, None, p=p_signal/p_ssum)
			signal = signal/G

		    else :
			signal = 0

		# get detector noise (photoelectrons)
		Nr = self.configs.detector_readout_noise

		if (Nr > 0) : noise = numpy.random.normal(0, Nr, None)
		else : noise = 0

		# the number of photons detected in the PMT
		PE = signal + noise

		intensity = [Photons, Exp, PE, int(PE)]

		return intensity


