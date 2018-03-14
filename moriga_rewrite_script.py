
import sys
import os
import numpy

from moriga_handler   import MorigaConfigs, MorigaVisualizer
from effects_handler import PhysicalEffects

def test_tirfm(t0, t1, nm, I0, Bg) :

	# create TIRF Microscopy (for Moriga-configuration)
	tirfm = MorigaConfigs()
	tirfm.set_Fluorophore(wave_length=532, intensity=I0, width=(110,400))
	tirfm.set_DichroicMirror('FF562-Di03-25x36')
	#tirfm.set_EmissionFilter('FF01-593_40-25')
	tirfm.set_Magnification(Mag=89)

	# Detector : CMOS Camera
	tirfm.set_Detector(detector='CMOS', image_size=(600,600), pixel_length=6.5e-6, \
			focal_point=(0.0,0.5,0.5), exposure_time=0.1, QE=0.73)
	tirfm.set_ADConverter(bit=16, offset=100, fullwell=30000)

	### Input data
	#tirfm.reset_InputData('/home/masaki/wrk/spatiocyte/models/data_moriga/model_%03dA/index_%03d' % (NNN,index), \
	#										start=t0, end=t1, observable="A")
	tirfm.reset_InputData('/home/masaki/wrk/spatiocyte/models/data_moriga/model_002A_%03dnm' % (nm), \
											start=t0, end=t1, observable="A")

	# create physical effects
	physics = PhysicalEffects()
	physics.set_background(mean=Bg)

	# create image and movie
	create = MorigaVisualizer(configs=tirfm, effects=physics)
	create.rewrite_InputData(output_file_dir='./data_moriga_01/csv/model_002A_%03dnm_%03ddeg' % (nm,deg))
	#create.output_frames(num_div=16)



if __name__ == "__main__":

        #t0 = int(sys.argv[1])
        #t1 = int(sys.argv[2])
        nm  = int(sys.argv[1])

	test_tirfm(0, 50, nm, I0, Bg)


