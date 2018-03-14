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
from epifm_handler import VisualizerError, EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

from scipy.special import j0
from scipy.misc    import toimage


class TIRFMConfigs(EPIFMConfigs) :

    '''

    TIRFM configration

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





class TIRFMVisualizer(EPIFMVisualizer) :

	'''
	TIRFM visualization class
	'''

	def __init__(self, configs=TIRFMConfigs(), effects=PhysicalEffects()) :

		assert isinstance(configs, TIRFMConfigs)
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



