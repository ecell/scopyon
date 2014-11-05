
import sys
import os
import numpy

from tirfm_handler   import TIRFMConfigs, TIRFMVisualizer
from effects_handler import PhysicalEffects

def test_tirfm(t0, t1, index=None) :

	# create TIRF Microscopy
	tirfm = TIRFMConfigs()
	tirfm.set_LightSource(source_type='LASER', wave_length=532, flux_density=10, angle=65.7)
	tirfm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
	#tirfm.set_Fluorophore(fluorophore_type='EGFP')
	#tirfm.set_Fluorophore(fluorophore_type='Gaussian', wave_length=532, intensity=0.55, width=(200,400))
	tirfm.set_DichroicMirror('FF562-Di03-25x36')
	tirfm.set_EmissionFilter('FF01-593_40-25')
	tirfm.set_Magnification(Mag=198)

	# Detector : CMOS Camera
#	tirfm.set_Detector(detector='CMOS', image_size=(600,600), pixel_length=6.5e-6, \
#			focal_point=(0.0,0.5,0.5), exposure_time=30e-3, QE=0.73)
#	tirfm.set_ADConverter(bit=16, offset=100, fullwell=30000)
	# Detector : EMCCD Camera
	tirfm.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, \
			focal_point=(0.0,0.5,0.5), exposure_time=30e-3, QE=0.92, readout_noise=50, emgain=300)
	tirfm.set_ADConverter(bit=14, offset=200, fullwell=800000)

	### Output data
	tirfm.set_OutputData(image_file_dir='./images')

	### Input data
	tirfm.set_InputData('./data/tmr_tirfm', start=t0, end=t1, observable="A")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	physics.set_Background(mean=10)

	# create image and movie
	create = TIRFMVisualizer(configs=tirfm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])

	test_tirfm(t0, t1)


