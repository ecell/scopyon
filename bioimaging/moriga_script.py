
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

	### Output data
	tirfm.set_OutputData(image_file_dir='/home/masaki/bioimaging_4public/data_moriga_02/numpys/model_002A_%03dnm_%02dI0_%02dBg_extfunc' % (nm,I0,Bg))

	### Input data
	#tirfm.set_InputData('./data_moriga_02/csv/model_002A_%03dnm' % (nm), start=t0, end=t1, observable="A")
	tirfm.set_InputData('/home/masaki/wrk/spatiocyte/models/data_moriga/model_002A_%03dnm' % (nm), start=t0, end=t1, observable="A")


	# create physical effects
	physics = PhysicalEffects()
	physics.set_background(mean=(0.458/0.75)*Bg)

	# create image and movie
	create = MorigaVisualizer(configs=tirfm, effects=physics)
	create.output_frames(num_div=16)



if __name__ == "__main__":

        t0 = float(sys.argv[1])
        t1 = float(sys.argv[2])
        nm = int(sys.argv[3])
        I0 = int(sys.argv[4])
        Bg = int(sys.argv[5])

	test_tirfm(t0, t1, nm, I0, Bg)


