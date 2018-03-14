
import sys
import os

from fcs_handler import FCSConfigs, FCSVisualizer
from effects_handler import PhysicalEffects

def test_fcs(t0, t1) :

	# create FCS
	fcs = FCSConfigs()

        fcs.set_LightSource(source_type='LASER', wave_length=488, flux=100e-6, radius=200e-9)
	#fcs.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	fcs.set_Fluorophore(fluorophore_type='EGFP')
	fcs.set_Pinhole(radius=28.8e-6)
	fcs.set_Magnification(Mag=60)

#	# PMT : Photon-counting mode
#	fcs.set_Detector(detector='PMT', mode="Photon-counting", focal_point=(0.3,0.5,0.5), \
#			pixel_length=210e-9, exposure_time=10e-6, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
#	fcs.set_ADConverter(bit=12, offset=0, fullwell=4096)

	# PMT : Anlalog mode
	fcs.set_Detector(detector='PMT', mode="Analog", focal_point=(0.3,0.5,0.5), \
			pixel_length=210e-9, exposure_time=10e-6, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
	fcs.set_ADConverter(bit=16, offset=0, fullwell=30000)

	### Output data
	#fcs.set_OutputData(image_file_dir='./images_fcs_D100_A100')
	#fcs.set_OutputData(image_file_dir='./images_fcs_D100_A200')
	#fcs.set_OutputData(image_file_dir='./images_fcs_D010_A100')
	#fcs.set_OutputData(image_file_dir='./images_fcs_D010_A200')
	fcs.set_OutputData(image_file_dir='./images_fcs_2StMD_A200')

	### Input data
	#fcs.set_InputData('/home/masaki/ecell3/latest/data/csv/beads_11', start=t0, end=t1, observable="A")
	#fcs.set_InputData('/home/masaki/ecell3/latest/data/csv/beads_13', start=t0, end=t1, observable="A")
	#fcs.set_InputData('/home/masaki/ecell3/latest/data/csv/beads_12', start=t0, end=t1, observable="A")
	#fcs.set_InputData('/home/masaki/ecell3/latest/data/csv/beads_14', start=t0, end=t1, observable="A")
	fcs.set_InputData('/home/masaki/ecell3/latest/data/csv/beads_15', start=t0, end=t1, observable="A")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=3)

	# create image and movie
	create = FCSVisualizer(configs=fcs, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])

	test_fcs(t0, t1)

