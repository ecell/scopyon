import sys
import os
import copy
import tempfile
import time
import math
import operator
import random
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
        self._set_data('source_angle', angle)

        print '--- Light Source :', self.source_type
        print '\tWave Length = ', self.source_wavelength, 'nm'
        print '\tBeam Flux Density = ', self.source_flux_density, 'W/cm2'
        print '\t1/e2 Radius = ', self.source_radius, 'm'
        print '\tAngle = ', self.source_angle, 'degree'




    def set_Illumination_path(self) :

        #r = self.radial
        #d = self.depth
	r = numpy.linspace(0, 20000, 20001)
	d = numpy.linspace(0, 20000, 20001)

        # (plank const) * (speed of light) [joules meter]
        hc = 2.00e-25

	# Illumination : Assume that uniform illumination (No gaussian)
	# flux density [W/cm2 (joules/sec/cm2)]
        P_0 = self.source_flux_density*1e+4

        # single photon energy
        wave_length = self.source_wavelength*1e-9
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
	theta_in = (self.source_angle/180.)*numpy.pi

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
        self.source_flux_density = numpy.array(map(lambda x : I_r*x, I_d))

	print 'Penetration depth :', depth, 'm'
	print 'Amplitude :', A2_Tp+A2_Ts
	print 'Photon Flux Density (Max) :', self.source_flux_density[0][0]



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

