
import sys
import os
import copy
import tempfile
import math
import operator
import random
#import h5py
import ctypes
import multiprocessing

#import pylab
import scipy
import numpy

import parameter_effects

from time import sleep

from scipy.special import j0
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.misc    import toimage

#from matplotlib.backends.backend_pdf import PdfPages

IMAGE_SIZE_LIMIT=3000


class PhysicalEffects() :

    '''
    Physical Effects setting class

	Linear conversion
	Background
	Photobleaching
	Photoblinking
    '''

    def __init__(self, user_configs_dict = None):

        # default setting
        configs_dict = parameter_effects.__dict__.copy()

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



    def set_Conversion(self, ratio = None) :

        print '--- Linear Conversion :'

        self._set_data('conversion_ratio', ratio)

        print '\tI_out / I_in = ', self.conversion_ratio



    def set_Background(self, mean = None) :

        print '--- Background :'

        self._set_data('background_switch', True)
        self._set_data('background_mean', mean)

        print '\tMean = ', self.background_mean, 'photons'



    def set_DetectorCrosstalk(self, width = None) :

        print '--- Detector Crosstalk :'

        self._set_data('detector_crosstalk_switch', True)
        self._set_data('detector_crosstalk_width', width)

        print '\tGaussian Width  = ', self.detector_crosstalk_width, 'pixels'



    def set_Photobleaching(self, rate = None) :

        print '--- Photobleaching :'
                                
        self._set_data('bleaching_switch', True)
        self._set_data('bleaching_rate', rate)
        
        print '\tPhotobleaching rate = ', self.bleaching_rate, '1/sec'



    def set_Photoblinking(self, P0 = None,
			alpha_on  = None,
			alpha_off = None
			) :

	print '--- Photoblinking :'

        self._set_data('blinking_switch', True)
        self._set_data('blinking_prob0', P0)
        self._set_data('blinking_alpha_on', alpha_on)
        self._set_data('blinking_alpha_off', alpha_off)

	print '\tP0 = ', self.blinking_prob0
	print '\talpha_on  = ', self.blinking_alpha_on
	print '\talpha_off = ', self.blinking_alpha_off



    def set_states(self, time, length) :

	# set photobleaching state
	if (self.bleaching_switch == True) :

	    array = multiprocessing.Array(ctypes.c_double, length)
	    self.bleaching_state = numpy.frombuffer(array.get_obj())

	    #init_value = self.get_bleaching(time)
	    self.bleaching_state[:] = numpy.array([1.0 for i in range(length)])


	# set photoblinking state
	if (self.blinking_switch == True) :

	    array0 = multiprocessing.Array(ctypes.c_double, length)
	    self.blinking_state = numpy.frombuffer(array0.get_obj())
	    self.blinking_state[:]  = map(lambda x : round(x), numpy.random.uniform(0, 1, length))

	    array1 = multiprocessing.Array(ctypes.c_double, length)
	    self.blinking_period = numpy.frombuffer(array1.get_obj())
	    self.blinking_period[:] = numpy.array([0.0 for i in range(length)])



    def get_Prob_blinking(self, state, time) :


	# blinking parameters
	P0 = self.blinking_prob0

	alpha_on  = self.blinking_alpha_on
	alpha_off = self.blinking_alpha_off

	# ON-state
	if (state) :
	    if (time > 0) :
		prob = P0/time**alpha_on
	    else :
		prob = P0

	# OFF-state
	else :
            if (time > 0) :
                prob = P0/time**alpha_off
            else :
                prob = P0

	return prob


