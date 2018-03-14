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

import hiroshima_configs
from epifm_handler import VisualizerError, EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

from ast import literal_eval
from scipy.special import j0
from scipy.misc    import toimage


class HiroshimaConfigs(EPIFMConfigs) :

    '''

    Hiroshima configration

	Low-angle oblique illumination using two incident beams
	    +
	EMCCD camera
    '''

    def __init__(self, user_configs_dict = None):

        # default setting
        configs_dict = hiroshima_configs.__dict__.copy()
        #configs_dict = parameter_configs.__dict__.copy()
        #configs_dict_tirfm = tirfm_configs.__dict__.copy()
        #configs_dict.update(configs_dict_tirfm)

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




    def set_LightSource(self,  source_type = None,
				wave_length = None,
                                flux_density = None,
                                radius = None,
                                angle  = None ) :

        self._set_data('source_switch', True)
        self._set_data('source_type', source_type)
        self._set_data('source_wavelength', wave_length)
        self._set_data('source_flux_density', flux_density)
        self._set_data('source_radius', radius)
        self._set_data('source_angle',  angle)
        #self._set_data('source_angle_left',  angle_left)
        #self._set_data('source_angle_right', angle_right)

        print('--- Light Source :', self.source_type)
        print('\tWave Length = ', self.source_wavelength, 'nm')
        print('\tBeam Flux Density = ', self.source_flux_density, 'W/cm2')
        print('\t1/e2 Radius = ', self.source_radius, 'm')
        print('\tAngle = ', self.source_angle, 'degree')
        #print '\tAngle (1) = ', self.source_angle_left, 'degree'
        #print '\tAngle (2) = ', self.source_angle_right, 'degree'



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
		    c_id = (float(data_id[1]), float(data_id[2]), float(data_id[3]))
		    # get Molecule ID and its state
		    m_id, s_id = literal_eval(data_id[5])
		    # get Fluorophore ID and compartment ID
		    f_id, l_id = literal_eval(data_id[6])

		    try :
		        p_state, cyc_id = float(data_id[7]), float(data_id[8])

		    except Exception :
		        obs = self.spatiocyte_index[int(s_id)]

			if (obs == 'R1' or obs == 'Rh1' or
			    obs == 'R2' or obs == 'Rh2' or obs == 'R3') :
		            p_state, cyc_id = 1.0, float('inf')
			elif (obs == 'rR' or obs == 'pR' or obs == 'rRh') :
		            p_state, cyc_id = 0.5, float('inf')
			elif (obs == 'rrR' or obs == 'rRR') :
		            p_state, cyc_id = 0.33333, float('inf')
			else :
		            p_state, cyc_id = 0.0, float('inf')

		    particles.append((c_id, m_id, s_id, l_id, p_state, cyc_id))
		
		data.append([time, particles])


            except Exception :
                print('Error : ', csv_file_path, ' not found')
		#exit()

	data.sort(lambda x, y:cmp(x[0], y[0]))

	# set data
        self._set_data('spatiocyte_data', data)

#	if len(self.spatiocyte_data) == 0:
#	    raise VisualizerError('Cannot find spatiocyte_data in any given csv files: ' \
#	                    + ', '.join(csv_file_directry))



class HiroshimaVisualizer(EPIFMVisualizer) :

	'''
	Hiroshima visualization class
	'''

	def __init__(self, configs=HiroshimaConfigs(), effects=PhysicalEffects()) :

		assert isinstance(configs, HiroshimaConfigs)
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



	def rewrite_InputFile(self, output_file_dir=None) :

		if not os.path.exists(output_file_dir):
		    os.makedirs(output_file_dir)

                # get focal point
                p_0 = self.get_focal_center()

                # beam position : Assuming beam position = focal point (for temporary)
                p_b = copy.copy(p_0)

		# set time, delta and count arrays
		time_array  = numpy.array(self.configs.shutter_time_array)
		delta_array = numpy.array(self.configs.shutter_delta_array)
		count_array = numpy.array(self.configs.shutter_count_array)

		# Snell's law
		amplitude0, penet_depth = self.snells_law(p_0, p_0)

		# get the number of emitted photons
		N_emit0 = self.get_emit_photons(amplitude0)

		# copy input-file
		csv_input = self.configs.spatiocyte_file_directry + '/pt-input.csv'
		shutil.copyfile(csv_input, output_file_dir + '/pt-input.csv')

		# copy shape-file
		csv_shape = self.configs.spatiocyte_file_directry + '/pt-shape.csv'
		shutil.copyfile(csv_shape, output_file_dir + '/pt-shape.csv')

		# get the total number of particles
		#N_particles = 12824
		N_particles = 13014

		# set molecule-states
		self.molecule_states = numpy.zeros(shape=(N_particles))

		# set fluorescence
		self.effects.set_photophysics_4epifm(time_array, delta_array, N_emit0, N_particles)

		for k in range(len(count_array)) :

		    # read input file
		    csv_file_path = self.configs.spatiocyte_file_directry + '/pt-%09d.0.csv' % (count_array[k])

		    csv_list = list(csv.reader(open(csv_file_path, 'r')))
		    dataset = numpy.array(csv_list)

		    # get molecular-states (unbound-bound)
		    self.set_molecular_states(k, dataset)

		    # set photobleaching-dataset arrays
		    self.set_photobleaching_dataset(k, dataset)

		    # get new-dataset arrays
		    new_dataset = self.get_new_dataset(k, N_emit0, dataset)

		    # reset arrays for photobleaching-state and photon-budget
		    self.reset_photobleaching_state(k, N_emit0)

		    # write output file
		    output_file = output_file_dir + '/pt-%09d.0.csv' % (count_array[k])

		    with open(output_file, 'w') as f :
		        writer = csv.writer(f)
		        writer.writerows(new_dataset)



        def set_molecular_states(self, count, dataset) :

		# get focal point
		p_0 = self.get_focal_center()

		# reset molecule-states
		self.molecule_states.fill(0)

		# set arrays for photobleaching-states and photon-budget
		keys = []
		vals = []

                # loop for particles
                for j, data_j in enumerate(dataset):

		    # set particle position
		    p_i = numpy.array(data_j[1:4]).astype('float')/1e-9

		    # Molecule ID and its state
		    m_id, s_id = literal_eval(data_j[5])
		    # Fluorophore ID and compartment ID
		    f_id, l_id = literal_eval(data_j[6])

		    # get fluorophre-ID
		    obs = self.configs.spatiocyte_index[int(s_id)]
		    self.molecule_states[m_id] = int(s_id)

		    if (obs == 'rR' or obs == 'pR') :
		        keys.append(m_id)
			vals.append(p_i)

		m_id = numpy.array(keys)
		data = numpy.array(vals)

		# set dict for internal-states
		internal_states = {}

		for j in range(len(m_id)) :

		    m_id0 = m_id[j]
		    p_j = data[j]

		    m_id_k = m_id[m_id != m_id0]
		    data_k = data[m_id != m_id0]
		    p_k = data_k

		    distance = numpy.array(map(lambda x : numpy.sqrt(numpy.sum((x-p_j)**2)), p_k))
		    index = distance.argmin()

		    m_id1 = m_id_k[index]

		    if not m_id0 in internal_states :
		        if not m_id1 in internal_states :

        		    # get photon-budget
        		    budget0 = self.effects.fluorescence_budget[m_id0]
        		    budget1 = self.effects.fluorescence_budget[m_id1]
        
        		    if (budget0 > 0 and budget1 > 0) :
        		        if (count == 0) :
        			    prob = numpy.random.uniform(0,1,1)

        		            if (prob > 0.5) : 
        		                internal_states[m_id0] = 1
        		                internal_states[m_id1] = 0
				    else :
        		                internal_states[m_id0] = 0
        		                internal_states[m_id1] = 1
        
        			else :
				    pasts = self.past_internal_states

				    if m_id0 in pasts :
        			        internal_states[m_id0] = pasts[m_id0]

				        if m_id1 in pasts :
        			            internal_states[m_id1] = pasts[m_id1]
					else :
        			            internal_states[m_id1] = int(numpy.abs(pasts[m_id0]-1))
				    else :
				        if m_id1 in pasts :
        			            internal_states[m_id1] = pasts[m_id1]
        			            internal_states[m_id0] = int(numpy.abs(pasts[m_id1]-1))
					else :
					    prob = numpy.random.uniform(0,1,1)

					    if (prob > 0.5) :
						internal_states[m_id0] = 1
						internal_states[m_id1] = 0
					    else :
						internal_states[m_id0] = 0
						internal_states[m_id1] = 1
        
        		    elif (budget0 > 0 and budget1 == 0) :
        			internal_states[m_id0] = 1
        			internal_states[m_id1] = 0
        
        		    elif (budget0 == 0 and budget1 > 0) :
        			internal_states[m_id0] = 0
        			internal_states[m_id1] = 1
        
        		    else :
        			internal_states[m_id0] = 0
        			internal_states[m_id1] = 0

		self.internal_states = internal_states



	def reset_photobleaching_state(self, count, N_emit0) :

		state_mo = self.molecule_states
		state_in = self.internal_states

		if (count > 0) :

		    # initialize the photobleaching-states and photon-budget
		    # (1) R  -> r  (shown as GFP)
		    # (2) rR -> rr (shown as GFP)
		    # (3) RR -> pR
		    molecules = set()

		    past_mo = self.past_molecule_states

		    case_flag = False

		    for i in range(len(state_mo)) :

		        obs0 = self.configs.spatiocyte_index[int(state_mo[i])]
		        obs1 = self.configs.spatiocyte_index[int(past_mo[i])]

		        if ((obs0 == 'GFP' and obs1 == 'R1') or
			    (obs0 == 'GFP' and obs1 == 'rR') or
			    (obs0 == 'GFP' and obs1 == 'rRh') or
			    (obs0 == 'GFP' and obs1 == 'Rh1')) :
			    case_flag = True

		        if ((obs0 == 'pR' and obs1 == 'R2') or
			    (obs0 == 'pR' and obs1 == 'Rh2')) :
			    if (state_in[i] == 0) :
			        case_flag = True

		        if (case_flag == True) :
			    molecules.add(i)
			    #print 'reset states :', count, '(', i, ':', obs1, '->', obs0, ')'

			case_flag = False

		    molecules = list(molecules)

    		    # initialize the photobleaching-states and photon-budget
		    if (len(molecules) > 0) :

                        if self.use_multiprocess():
    		            # set arrays for photobleaching-state and photon-budget
                            state_pb = {}
                            budget = {}
    
                            num_processes = min(multiprocessing.cpu_count(), len(molecules))
                            n, m = divmod(len(molecules), num_processes)
    
                            chunks = [n + 1 if i < m else n for i in range(num_processes)]
    
                            processes = []
                            start_index = 0
    
                            for chunk in chunks:
                                stop_index = start_index + chunk
                                p, c = multiprocessing.Pipe()
                                process = multiprocessing.Process(target=self.get_state_initialization_process,
								args=(molecules[start_index:stop_index], N_emit0, c))
                                processes.append((p, process))
                                start_index = stop_index
    
                            for _, process in processes:
                                process.start()
    
                            for pipe, process in processes:
    			        new_state_pb, new_budget = pipe.recv()
    			        state_pb.update(new_state_pb)
    			        budget.update(new_budget)
                                process.join()
    
                        else :
                            state_pb, budget = self.get_state_initialization(molecules, N_emit0)
    
    		        # reset global-arrays for photobleaching-state and photon-budget
    		        for key, value in state_pb.items() :
    
    		            self.effects.fluorescence_state[key] = state_pb[key]
    		            self.effects.fluorescence_budget[key] = budget[key]

		# set past-states
		self.past_molecule_states = copy.deepcopy(state_mo)
		self.past_internal_states = copy.deepcopy(state_in)



        def get_state_initialization(self, molecules, N_emit0) :

		state_pb = self.effects.fluorescence_state
		budget = self.effects.fluorescence_budget

		# get delta-time array
		delta = numpy.array(self.configs.shutter_delta_array)

		# set dicts for resetting states and photon-budget
		result_state_pb = {}
		result_budget = {}

		for i in molecules :

		    # reset photobleaching-time and photon-budget
		    tau_i, budget_i = self.effects.get_photobleaching_property(delta[0], N_emit0)

		    # reset the photobleaching-state
		    state_pb_i = numpy.zeros(shape=(len(state_pb[i])))

	    	    Ni = (numpy.abs(numpy.cumsum(delta) - tau_i)).argmin()
		    state_pb_i[0:Ni] = numpy.ones(shape=(Ni))

		    result_state_pb[i] = state_pb_i
		    result_budget[i] = budget_i

		return result_state_pb, result_budget



        def get_state_initialization_process(self, molecules, N_emit0, pipe) :

        	pipe.send(self.get_state_initialization(molecules, N_emit0))



        def get_photobleaching_dataset(self, count, dataset) :

		# get focal point
		p_0 = self.get_focal_center()

		# set arrays for photobleaching-states and photon-budget
		result_state_pb = {}
		result_budget = {}

                # loop for particles
                for j, data_j in enumerate(dataset):

		    # set particle position
		    p_i = numpy.array(data_j[1:4]).astype('float')/1e-9

		    # Snell's law
		    amplitude, penet_depth = self.snells_law(p_i, p_0)

		    # particle coordinate in real(nm) scale
		    p_i, radial, depth = self.get_coordinate(p_i, p_0)

		    # Molecule ID and its state
		    m_id, s_id = literal_eval(data_j[5])
		    # Fluorophore ID and compartment ID
		    f_id, l_id = literal_eval(data_j[6])

		    # get molecule-state
		    obs = self.configs.spatiocyte_index[int(s_id)]
 
		    if (obs == 'R1' or obs == 'Rh1' or
			obs == 'R2' or obs == 'Rh2' or obs == 'R3') :
			state_j = 1

		    elif (obs == 'rR' or obs == 'pR' or obs == 'rRh') :
			state_j = self.internal_states[m_id]
		    else :
			state_j = 0

		    # get exponential amplitude (only for observation at basal surface)
		    #amplitide = amplitude*numpy.exp(-depth/pent_depth)

		    # get the number of emitted photons
		    N_emit = self.get_emit_photons(amplitude)

		    # get global-arrays for photobleaching-state and photon-budget
		    state_pb = self.effects.fluorescence_state[m_id,count]
		    budget = self.effects.fluorescence_budget[m_id]

		    # reset photon-budget
		    photons = budget - N_emit*state_j

		    if (photons > 0) :
		        budget = photons
			state_pb = state_j
		    else :
		        budget = 0
			state_pb = 0

		    result_state_pb[m_id] = state_pb
		    result_budget[m_id] = budget

		return result_state_pb, result_budget



#	def get_molecule_plane(self, cell, data, p_b, p_0) :
#
#		# particles coordinate, species and lattice IDs
#                c_id, s_id, l_id, p_state, cyc_id = data
#                
#		if self.configs.spatiocyte_observables[s_index] is True :
#
#		    if (p_state > 0) :
#
#			p_i = numpy.array(c_id)/1e-9
#
#			# Snell's law
#			amplitude, penet_depth = self.snells_law(p_i, p_0)
#
#			#angle_left  = self.configs.source_angle_left
#			#angle_right = self.configs.source_angle_right
#			#A2_left,  penet_depth_left  = self.snells_law(p_i, p_0, angle_left)
#			#A2_right, penet_depth_right = self.snells_law(p_i, p_0, angle_right)
#			#amplitude = A2_left + A2_right
#
#		        # particles coordinate in real(nm) scale
#                        p_i, radial, depth = self.get_coordinate(p_i, p_0)
#
#			# reset p_state
#			obs = self.configs.spatiocyte_index[int(s_id)]
#
#			if (obs == 'R1' or obs == 'Rh1' or
#			    obs == 'R2' or obs == 'Rh2') :
#			    p_state = 1.0
#			elif (obs == 'rR' or obs == 'pR') :
#			    p_state = 0.5
#			else :
#			    p_state = 0.0
#
#                        # get signal matrix
#                        signal = self.get_signal(amplitude, radial, depth, p_state)
#
#                        # add signal matrix to image plane
#                        self.overwrite_signal(cell, signal, p_i)
#
#
#
