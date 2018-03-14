import sys
import os
import numpy

from minDE_handler   import EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

def test_minDE(t0, t1) :

	# create EPIFM imaging
	minDE = EPIFMConfigs()
	minDE.set_Shutter(start_time=t0, end_time=t1)
	minDE.set_LightSource(source_type='LASER', wave_length=532, flux_density=20, angle=72)
	minDE.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0)
	minDE.set_DichroicMirror('FF562-Di03-25x36')
	minDE.set_Magnification(Mag=100)

	# Detector : EMCCD Camera
	minDE.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, exposure_time=0.200, \
				focal_point=(0.0,0.5,0.5), QE=0.92, readout_noise=100, emgain=300)
	minDE.set_ADConverter(bit=16, offset=2000, fullwell=800000)

	# Output data
	#minDE.set_OutputData(image_file_dir='/home/masaki/bioimaging_4public/data_for_satya/numpys/minE')
	minDE.set_OutputData(image_file_dir='/home/masaki/bioimaging_4public/data_for_satya/numpys/minD')

	# Input data
	csv_dir = '/home/masaki/wrk/spatiocyte/models/Satya/data_minDE'
	#minDE.set_InputFile(csv_dir, observable="EE")
	minDE.set_InputFile(csv_dir, observable="D")
	minDE.set_ShapeFile(csv_dir)

	# create physical effects
	physics = PhysicalEffects()
	physics.set_background(mean=0.01)
	physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
	#physics.set_photobleaching(tau0=2.27, alpha=0.73)

	# create image
	create = EPIFMVisualizer(configs=minDE, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])

	test_minDE(t0, t1)


