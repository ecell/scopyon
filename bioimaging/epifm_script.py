import sys
import os
import numpy

from epifm_handler   import EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

def test_epifm(t0, t1, angle=0) :

	# create EPIFM imaging
	epifm = EPIFMConfigs()
	epifm.set_LightSource(source_type='LASER', wave_length=561, flux_density=20, angle=angle)
	epifm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	epifm.set_DichroicMirror('FF562-Di03-25x36')
	epifm.set_EmissionFilter('FF01-593_40-25')
	epifm.set_Magnification(Mag=250)

#	# Detector : CMOS Camera
#	epifm.set_Detector(detector='CMOS', image_size=(600,600), pixel_length=6.5e-6, \
#				exposure_time=0.5, QE=0.73)
#	epifm.set_ADConverter(bit=16, offset=100, fullwell=30000)

	# Detector : EMCCD Camera
	epifm.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, exposure_time=0.1, \
				focal_point=(0.8355,0.5, 0.5), QE=0.92, readout_noise=100, emgain=300)
	epifm.set_ADConverter(bit=16, offset=2000, fullwell=800000)

	# Output data
	epifm.set_OutputData(image_file_dir='./numpys_epifm_TEST')
	#epifm.set_OutputData(image_file_dir='./numpys_epifm_TEST_%02ddeg' % (angle))
	#epifm.set_OutputData(image_file_dir='./numpys_epifm_TEST_08deg_bleach')

	# Input data
	csv_dir = '/home/masaki/wrk/spatiocyte/models/Hiroshima/data_HRG-ErbB/SM/csv/TEST'
	#csv_dir = '/home/masaki/bioimaging_4public/data_epifm_TEST_08deg_bleach'
	epifm.set_InputData(csv_dir, start=t0, end=t1, observable="r1")
	epifm.set_ShapeData(csv_dir)

	# create physical effects
	physics = PhysicalEffects()
	physics.set_background(mean=0)
	physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
	#physics.set_photobleaching(tau0=1.8, alpha=0.73)
	#physics.set_photoblinking(t0_on=1.00, a_on=0.58, t0_off=10e-6, a_off=0.48)

	# create image
	create = EPIFMVisualizer(configs=epifm, effects=physics)
	#create.rewrite_InputData(output_file_dir='./data_epifm_01000A')
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])
        angle = float(sys.argv[3])

	test_epifm(t0, t1, angle)


