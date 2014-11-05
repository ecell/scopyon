
import sys
import os

from pointscan_confm_handler import PointScanConfocalConfigs, PointScanConfocalVisualizer
from effects_handler import PhysicalEffects

def pten_confm(t0, t1) :

	# create Point-scanning Confocal Microscopy
	confm = PointScanConfocalConfigs()

        confm.set_LightSource(source_type='LASER', wave_length=561, flux=100e-6, radius=200e-9)
	confm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	confm.set_Pinhole(radius=20.0e-6)
	confm.set_Magnification(Mag=60)

	# PMT : Anlalog mode
	confm.set_Detector(detector='PMT', mode="Analog", image_size=(512,512), focal_point=(0.4,0.5,0.5),
			pixel_length=414.3e-9, scan_time=5.00, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
	confm.set_ADConverter(bit=12, offset=0, fullwell=4096)
	confm.set_OutputData(image_file_dir='./images_pten')

	# Input data : EGF model file
	confm.set_InputData('./data/csv/pten', start=t0, end=t1, observable="PTEN")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=20)
	#physics.set_DetectorCrosstalk(width=1.00)

	# create image and movie
	create = PointScanConfocalVisualizer(configs=confm, effects=physics)
	create.output_frames(num_div=16)



def pip3_confm(t0, t1) :

	# create Point-scanning Confocal Microscopy
	confm = PointScanConfocalConfigs()

        confm.set_LightSource(source_type='LASER', wave_length=488, flux=100e-6, radius=200e-9)
	confm.set_Fluorophore(fluorophore_type='EGFP')
	confm.set_Pinhole(radius=20e-6)
	confm.set_Magnification(Mag=60)

	# PMT : Anlalog mode
	confm.set_Detector(detector='PMT', mode="Analog", image_size=(512,512), focal_point=(0.4,0.5,0.5),
			pixel_length=414.3e-9, scan_time=5.00, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
	confm.set_ADConverter(bit=12, offset=0, fullwell=4096)
	confm.set_OutputData(image_file_dir='./images_pip3')

	# Input data : EGF model file
	confm.set_InputData('./data/pten', start=t0, end=t1, observable="PIP3")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=20)
	#physics.set_DetectorCrosstalk(width=1.00)

	# create image and movie
	create = PointScanConfocalVisualizer(configs=confm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        index = sys.argv[1]
        t0 = float(sys.argv[2])
        t1 = float(sys.argv[3])

	if (index == "PTEN") :
	    pten_confm(t0, t1)
	elif (index == "PIP3") :
	    pip3_confm(t0, t1)

