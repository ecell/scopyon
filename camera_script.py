
import sys
import os

from camera_handler   import CameraConfigs, CameraVisualizer

signal = [i for i in range(11)]

def test_camera(index) :

	# create Camera configuration
	camera = CameraConfigs()

	# Detector : CMOS Camera
	#camera.set_Detector(detector='CMOS', image_size=(100,100), pixel_length=6.5e-6, \
	#		focal_point=(0.0,0.5,0.5), exposure_time=30e-3, QE=0.73)
	#camera.set_ADConverter(bit=16, offset=100, fullwell=30000)

	# Detector : EMCCD Camera
	camera.set_Detector(detector='EMCCD', image_size=(100,100), pixel_length=16e-6, \
			focal_point=(0.0,0.5,0.5), exposure_time=30e-3, QE=0.92, readout_noise=100, emgain=1)
	camera.set_ADConverter(bit=16, offset=2000, fullwell=370000)

	### Output data
	camera.set_OutputData(image_file_dir='./images_camera_emccd_bg05_x001')

	# create image and movie
	create = CameraVisualizer(configs=camera)
	create.output_frames(index=index, signal=signal[index], background=5)



if __name__ == "__main__":

	index = int(sys.argv[1])

	test_camera(index)


