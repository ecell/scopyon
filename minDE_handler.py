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

import multiprocessing

import scipy
import numpy

import parameter_configs
from epifm_handler import VisualizerError, EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

from ast import literal_eval
from scipy.special import j0
from scipy.misc    import toimage


class MinDEConfigs(EPIFMConfigs) :

    '''

    MinDE configration

	Low-angle oblique illumination using two incident beams
	    +
	EMCCD camera
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




    def set_Spatiocyte_data_arrays(self) :

	# get spatiocyte file directry
	csv_file_directry = self.spatiocyte_file_directry

	# set data-array
	data = []

	# get count-array
	count_array = numpy.array(self.shutter_count_array)

	# read lattice file
	for i in range(len(count_array)) :

	    csv_file_path = csv_file_directry + '/pt-%09d.0.csv' % (count_array[i])
            
            try :

                csv_file = open(csv_file_path, 'r')

		dataset = []

		for row in csv.reader(csv_file) :
		    dataset.append(row)

                # get Time
		time = float(dataset[0][0])

		particles = []

		for data_id in dataset :
		    # get Coordinate
		    #c_id = (float(data_id[1]), float(data_id[2]), float(data_id[3]))
		    c_id = (float(data_id[3]), float(data_id[2]), float(data_id[1]))
		    # get Molecule ID and its state
		    m_id, s_id = literal_eval(data_id[5])
		    # get Fluorophore ID and compartment ID
		    f_id, l_id = literal_eval(data_id[6])

		    try :
		        p_state, cyc_id = float(data_id[7]), float(data_id[8])

		    except Exception :
		        p_state, cyc_id = 1.0, float('inf')

		    particles.append((c_id, s_id, l_id, p_state, cyc_id))
		
		data.append([time, particles])


            except Exception :
                print 'Warning : ', csv_file_path, ' not found'
		#exit()

	data.sort(lambda x, y:cmp(x[0], y[0]))

	# set data
        self._set_data('spatiocyte_data', data)

#	if len(self.spatiocyte_data) == 0:
#	    raise VisualizerError('Cannot find spatiocyte_data in any given csv files: ' \
#	                    + ', '.join(csv_file_directry))



class MinDEVisualizer(EPIFMVisualizer) :

	'''
	MinDE visualization class
	'''

	def __init__(self, configs=MinDEConfigs(), effects=PhysicalEffects()) :

		assert isinstance(configs, MinDEConfigs)
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



