
import sys
import os

from epifm_handler import EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

def test_epifm(t0, t1) :

	# create EPIF Microscopy
	epifm = EPIFMConfigs()

	epifm.set_LightSource(source_type='LASER', wave_length=532, power=20e-3, radius=20e-6)
	epifm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	#epifm.set_Fluorophore(fluorophore_type='Gaussian', wave_length=578, width=(70.0, 140.0))
	epifm.set_DichroicMirror('FF562-Di03-25x36')
	epifm.set_EmissionFilter('FF01-593_40-25')
	epifm.set_Magnification(Mag=336)
	epifm.set_Detector(detector='EMCCD', zoom=1, emgain=1, focal_point=(0.3,0.5,0.5), exposure_time=30e-3)
	epifm.set_ADConverter(bit=16, offset=2000, fullwell=370000) # for EMCCD
	#epifm.set_OutputData(image_file_dir='./images_dicty_02_epifm_zaxis09')
	epifm.set_OutputData(image_file_dir='./images_test')
	epifm.set_InputData('/home/masaki/ecell3/latest/data/csv/simple_dicty_02', start=t0, end=t1)

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=30)
	physics.set_DetectorCrosstalk(width=1.00) # for EMCCD

	# create image and movie
	create = EPIFMVisualizer(configs=epifm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

	t0 = float(sys.argv[1])
	t1 = float(sys.argv[2])

	test_epifm(t0, t1)


