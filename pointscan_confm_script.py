
import sys
import os

from pointscan_confm_handler import PointScanConfocalConfigs, PointScanConfocalVisualizer
from effects_handler import PhysicalEffects

def test_confm(t0, t1, index=None) :

	# create Point-scanning Confocal Microscopy
	confm = PointScanConfocalConfigs()

        confm.set_LightSource(source_type='LASER', wave_length=532, flux=10e-6, radius=400e-9)
	#confm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	confm.set_Fluorophore(fluorophore_type='EGFP')
	confm.set_Pinhole(radius=90e-6)
	confm.set_Magnification(Mag=60)

#	# PMT : Photon-counting mode
#	confm.set_Detector(detector='PMT', mode="Photon-counting", image_size=(1024,1024), focal_point=(0.4,0.5,0.5), \
#			pixel_length=207.16e-9, scan_time=1.15, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
#	confm.set_ADConverter(bit=12, offset=0, fullwell=4096)
#	confm.set_OutputData(image_file_dir='./images_erk102_pulses')

	# PMT : Anlalog mode
	confm.set_Detector(detector='PMT', mode="Analog", image_size=(1024,1024), focal_point=(0.4,0.5,0.5), \
			pixel_length=207.16e-9, scan_time=1.15, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
	confm.set_ADConverter(bit=12, offset=0, fullwell=4096)
	confm.set_OutputData(image_file_dir='./images')

	# Input data : EGF model file
	confm.set_InputData('./data/tmr_lscm', start=t0, end=t1, observable="A")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=10)
	#physics.set_DetectorCrosstalk(width=1.00)

	# create image and movie
	create = PointScanConfocalVisualizer(configs=confm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])

	test_confm(t0, t1)

