import sys
import os
import numpy

from epifm_handler   import EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

def test_epifm(t0, t1) :

	# create EPIFM imaging
	epifm = EPIFMConfigs()
	epifm.set_Shutter(start_time=t0, end_time=t1)
	epifm.set_LightSource(source_type='LASER', wave_length=532, flux_density=20, angle=72)
	epifm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0)
	epifm.set_DichroicMirror('FF562-Di03-25x36')
	epifm.set_Magnification(Mag=364)

	# Detector : EMCCD Camera
	epifm.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, exposure_time=0.150, \
				focal_point=(0.0,0.5,0.5), QE=0.92, readout_noise=100, emgain=300)
	epifm.set_ADConverter(bit=16, offset=2000, fullwell=800000)

	# Output data
	#epifm.set_OutputData(image_file_dir='./data_kaizu/numpys/numpys_ST_030ms')
	epifm.set_OutputData(image_file_dir='./data_kaizu/numpys/numpys_S2_150ms')

	# Input data
	#csv_dir = './data_kaizu/data_pb/ST_030ms'
	csv_dir = './data_kaizu/data_pb/S2_150ms'
	#csv_dir = '/home/masaki/wrk/spatiocyte/models/Kaizu/data/ST'
	epifm.set_InputFile(csv_dir, observable="S")
	epifm.set_ShapeFile(csv_dir)

	# create physical effects
	physics = PhysicalEffects()
	physics.set_background(mean=0.01)
	physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
	physics.set_photobleaching(tau0=2.27, alpha=0.73)

	# create image
	create = EPIFMVisualizer(configs=epifm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])

	test_epifm(t0, t1)


