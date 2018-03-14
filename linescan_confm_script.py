
import sys
import os

from linescan_confm_handler import LineScanConfocalConfigs, LineScanConfocalVisualizer
from effects_handler import PhysicalEffects

def test_confm(t0, t1) :

	# create Line-scanning Confocal Microscopy
	confm = LineScanConfocalConfigs()

        confm.set_LightSource(source_type='LASER', wave_length=532, power=20e-3, radius=200e-9)
	confm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	#confm.set_Fluorophore(fluorophore_type='Gaussian', wave_length=508, width=(70.0, 140.0))
	#confm.set_DichroicMirror('FF562-Di03-25x36')
	#confm.set_EmissionFilter('FF01-593_40-25')
	confm.set_Slits(size=16e-6)
	confm.set_Magnification(Mag=160)
	#confm.set_Detector(detector='EMCCD', zoom=1, emgain=1, focal_point=(0.9,0.5,0.5), exposure_time=50e-3)
	confm.set_Detector(detector='EMCCD', zoom=1, emgain=100, focal_point=(0.8,0.5,0.5), exposure_time=100)
	confm.set_ADConverter(bit=16, offset=2000, fullwell=370000)
	#confm.set_OutputData(image_file_dir='./images_dicty_02_line_zaxis09')
	confm.set_OutputData(image_file_dir='./images_pten_zaxis09')
	#confm.set_InputData('/home/masaki/ecell3/latest/data/csv/simple_dicty_02', start=t0, end=t1)
	confm.set_InputData('/home/masaki/ecell3/latest/data/csv/pten', start=t0, end=t1, observable="PTEN")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=30)
	physics.set_DetectorCrosstalk(width=1.00)

	# create image and movie
	create = LineScanConfocalVisualizer(configs=confm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])

	test_confm(t0, t1)

