import sys
import os
import numpy

from tirfm_handler   import TIRFMConfigs, TIRFMVisualizer
from effects_handler import PhysicalEffects

def test_tirfm(t0, t1) :

	# create TIRFM imaging
	tirfm = TIRFMConfigs()
	tirfm.set_LightSource(source_type='LASER', wave_length=561, flux_density=20, angle=65.7)
	tirfm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	tirfm.set_DichroicMirror('FF562-Di03-25x36')
	tirfm.set_EmissionFilter('FF01-593_40-25')
	tirfm.set_Magnification(Mag=250)

#	# Detector : CMOS Camera
#	tirfm.set_Detector(detector='CMOS', image_size=(600,600), pixel_length=6.5e-6, \
#			focal_point=(0.0,0.5,0.5), exposure_time=0.5, QE=0.73)
#	tirfm.set_ADConverter(bit=16, offset=100, fullwell=30000)

	# Detector : EMCCD Camera
	tirfm.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, \
			focal_point=(0.0,0.5,0.5), exposure_time=0.1, QE=0.92, readout_noise=100, emgain=300)
	tirfm.set_ADConverter(bit=16, offset=2000, fullwell=800000)

#	# Output data
#	tirfm.set_OutputData(image_file_dir='./numpys_tirfm_01000A')

	# Input data
	tirfm.reset_InputData('/home/masaki/wrk/spatiocyte/models/Hiroshima/data_HRG-ErbB/SM/csv/TEST', start=t0, end=t1, observable="R")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_background(mean=0)
	physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
	physics.set_photobleaching(tau0=1.8, alpha=0.73)
	#physics.set_photoblinking(t0_on=1.00, a_on=0.58, t0_off=10e-6, a_off=0.48)

	# create image
	create = TIRFMVisualizer(configs=tirfm, effects=physics)
	create.rewrite_InputData(output_file_dir='./data_tirfm_TEST_bleach')
	#create.rewrite_InputData(output_file_dir='./data_tirfm_TEST_blink')
	#create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])

	test_tirfm(t0, t1)


