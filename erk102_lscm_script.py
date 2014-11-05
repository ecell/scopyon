
import sys
import os

from pointscan_confm_handler import PointScanConfocalConfigs, PointScanConfocalVisualizer
from effects_handler import PhysicalEffects

def confm_pulses(t0, t1) :

	# create Point-scanning Confocal Microscopy
	confm = PointScanConfocalConfigs()

        confm.set_LightSource(source_type='LASER', wave_length=488, flux=100e-6, radius=200e-9)
	confm.set_Fluorophore(fluorophore_type='EGFP')
	confm.set_Pinhole(radius=28.8e-6)
	confm.set_Magnification(Mag=60)

	# PMT : Analog mode
	confm.set_Detector(detector='PMT', mode="Analog", image_size=(1024,1024), focal_point=(0.3,0.5,0.5), \
			pixel_length=207.16e-9, scan_time=1.15, QE=0.3, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
	confm.set_ADConverter(bit=12, offset=0, fullwell=4096)
	confm.set_OutputData(image_file_dir='./images_erk102')

	# Input data : EGF model file
	confm.set_InputData('./data/erk102', start=t0, end=t1, observable="ERK")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=5)
	#physics.set_DetectorCrosstalk(width=1.00)

	# create image and movie
	create = PointScanConfocalVisualizer(configs=confm, effects=physics)
	create.output_frames(num_div=16)



def confm_analog(t0, t1) :

	# create Point-scanning Confocal Microscopy
	confm = PointScanConfocalConfigs()

        confm.set_LightSource(source_type='LASER', wave_length=488, flux=100e-6, radius=200e-9)
	confm.set_Fluorophore(fluorophore_type='EGFP')
	confm.set_Pinhole(radius=28.8e-6)
	confm.set_Magnification(Mag=60)

	# PMT : Anlalog mode
	confm.set_Detector(detector='PMT', mode="Analog", image_size=(1024,1024), focal_point=(0.3,0.5,0.5),
			pixel_length=207.16e-9, scan_time=1.15, QE=0.3, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
	confm.set_ADConverter(bit=12, offset=0, fullwell=4096)
	confm.set_OutputData(image_file_dir='./images_erk102_mode_analog_100uW')

	# Input data : EGF model file
	confm.set_InputData('/home/masaki/ecell3/latest/data/csv/erk102', start=t0, end=t1, observable="ERK")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=5)
	#physics.set_DetectorCrosstalk(width=1.00)

	# create image and movie
	create = PointScanConfocalVisualizer(configs=confm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        index = sys.argv[1]
        t0 = float(sys.argv[2])
        t1 = float(sys.argv[3])

	if (index == "Photon-counting") :
	    confm_pulses(t0, t1)

	if (index == "Analog") :
	    confm_analog(t0, t1)

