
import sys
import os

from pmt_handler   import PMTConfigs, PMTVisualizer

flux = [0] + [10**i for i in range(10)]

def test_pmt(index) :

	# create PMT configuration
	pmt = PMTConfigs()
	pmt.set_Detector(detector='PMT', mode="Analog", readout_noise=0, dark_count=0, \
			bandwidth=5e+3, QE=0.30, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)

	### Output data
	#pmt.set_OutputData(image_file_dir='./images_pmt_pulses_1000dc')
	#pmt.set_OutputData(image_file_dir='./images_pmt_analog_1000dc')
	pmt.set_OutputData(image_file_dir='./images_pmt_test')

	# create image and movie
	create = PMTVisualizer(configs=pmt)
	create.output_frames(index=index, signal=flux[index], background=0, duration=0.1)
	#create.output_frames(index=index, signal=flux[index], background=0, image_size=(100,100))



if __name__ == "__main__":

	index = int(sys.argv[1])

	test_pmt(index)


