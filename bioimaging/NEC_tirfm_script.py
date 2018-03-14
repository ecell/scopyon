"""
    test_script.py:

    User script to create the image from the simulated EPI-TIR fluorescence Microscopy (EPI-TIRFM)
"""
import sys
import os
import numpy

from tirfm_handler   import TIRFMConfigs, TIRFMVisualizer
from effects_handler import PhysicalEffects

def test_tirfm(t0, t1, beam, dist=None) :

	# create TIRF Microscopy
	tirfm = TIRFMConfigs()
	tirfm.set_LightSource(source_type='LASER', wave_length=532, flux_density=beam, angle=65.7)
	tirfm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	tirfm.set_DichroicMirror('FF562-Di03-25x36')
	tirfm.set_EmissionFilter('FF01-593_40-25')
	tirfm.set_Magnification(Mag=100)
	#tirfm.set_Magnification(Mag=250)

	# Detector : CMOS Camera
	tirfm.set_Detector(detector='CMOS', image_size=(600,600), pixel_length=6.5e-6, \
			focal_point=(0.0,0.5,0.5), exposure_time=30e-3, QE=0.73)
	tirfm.set_ADConverter(bit=16, offset=100, fullwell=30000)
#	# Detector : EMCCD Camera
#	tirfm.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, \
#			focal_point=(0.0,0.5,0.5), exposure_time=30e-3, QE=0.92, readout_noise=100, emgain=300)
#	tirfm.set_ADConverter(bit=16, offset=2000, fullwell=370000)

	### Output data
	#tifm.set_OutputData(image_file_dir='./images')
	#tirfm.set_OutputData(image_file_dir='./numpys_test')
	#tirfm.set_OutputData(image_file_dir='./numpys_nec/numpys_nec_2A_%02dw_cmos_%03dnm' % (beam, dist))
	#tirfm.set_OutputData(image_file_dir='./numpys_nec/numpys_nec_2A_%02dw_emccd_%03dnm' % (beam, dist))
	#tirfm.set_OutputData(image_file_dir='./numpys_nec/numpys_nec_100A_%02dw_cmos' % (beam))
	#tirfm.set_OutputData(image_file_dir='./numpys_nec/numpys_nec_100A_%02dw_emccd' % (beam))
	tirfm.set_OutputData(image_file_dir='./numpys_nec_03/numpys_nec_05000A_20w_cmos')

	### Input data
	#tirfm.set_InputData('/home/masaki/ecell3/latest/data/csv/beads_nec_2A', start=t0, end=t1, dist_nm=dist)
	tirfm.set_InputData('/home/masaki/ecell3/latest/data/csv/beads_nec_05000A', start=t0, end=t1)

	# create physical effects
	physics = PhysicalEffects()
	physics.set_background(mean=0.1)
	physics.set_fluorescence(quantum_yield=1.00, abs_coefficient=100000)
	#physics.set_photobleaching(tau0=1.8, alpha=0.73)
	#physics.set_photoactivation(turn_on_ratio=1000, activation_yield=0.1, frac_preactivation=0.00)

	# create image and movie
	create = TIRFMVisualizer(configs=tirfm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])
        beam = 20#float(sys.argv[3])
        dist = 200#float(sys.argv[4])

	test_tirfm(t0, t1, beam, dist)


