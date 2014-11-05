
import sys
from fcs_handler import FCSConfigs, FCSVisualizer
from effects_handler import PhysicalEffects

def test_fcs(t0, t1) :

	# create FCS
	fcs = FCSConfigs()

	fcs.set_LightSource(source_type='LASER', wave_length=473, power=100e-6, radius=200e-9)
	fcs.set_Fluorophore(fluorophore_type='Qdot 605')
	#fcs.set_Fluorophore(fluorophore_type='Gaussian', wave_length=605, width=(100.0, 200.0))
	fcs.set_Magnification(Mag=336)
	fcs.set_Pinhole(radius=16e-6)
	fcs.set_Detector(detector='PMT', zoom=1, emgain=1e+6, focal_point=(0.5,0.5,0.5), bandwidth=50e+3, mode='Pulse')
	fcs.set_ADConverter(bit=16, offset=2000, fullwell=20000) # for Pulse
	fcs.set_OutputData(image_file_dir='./output_fcs_0001')
	fcs.set_InputData('/home/masaki/ecell3/latest/data/csv/test_fcs_0001', start=t0, end=t1)

	# create physical effects
	physics = PhysicalEffects()
	physics.set_Conversion(ratio=1e-6)
	#physics.set_Background(mean=30)

	# create image and movie
	create = FCSVisualizer(configs=fcs, effects=physics)
	create.output_frames(num_div=1)


if __name__ == "__main__":

	t0  = float(sys.argv[1])
	t1  = float(sys.argv[2])

	test_fcs(t0, t1)

