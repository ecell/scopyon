import sys
import os
import numpy

from epifm_handler   import EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

def test_repressilator(t0, t1) :

    # create TIRFM imaging
    repressilator = EPIFMConfigs()
    repressilator.set_Shutter(start_time=t0, end_time=t1)
    #repressilator.set_Shutter(start_time=t0, end_time=t1, time_open=0.150, time_lapse=20)
    repressilator.set_LightSource(source_type='LASER', wave_length=532, flux_density=20, angle=72)
    repressilator.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0)
    repressilator.set_DichroicMirror('FF562-Di03-25x36')
    #repressilator.set_Magnification(Mag=234) # EMCCD
    repressilator.set_Magnification(Mag=95) # CMOS

#       # Detector : EMCCD Camera
#       repressilator.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, exposure_time=0.050, \
#                               focal_point=(0.0,0.5,0.5), QE=0.92, readout_noise=100, emgain=300)
#       repressilator.set_ADConverter(bit=16, offset=2000, fullwell=800000)
    # Detector : CMOS Camera
    repressilator.set_Detector(detector='CMOS', image_size=(512,512), pixel_length=6.5e-6, \
                            exposure_time=0.05, focal_point=(0.0,0.5,0.5), QE=0.73)
    repressilator.set_ADConverter(bit=16, offset=100, fullwell=30000, fpn_type='column', fpn_count=10)

    # Output data
    #repressilator.set_OutputData(image_file_dir='./data_for_nikon/numpys/repressilator02_emccd')
    repressilator.set_OutputData(image_file_dir='./data_for_nikon/numpys/repressilator02_cmos')

    # Input data
    csv_dir = '/home/masaki/wrk/spatiocyte/models/Nikon/data/repressilator00'
    repressilator.set_InputFile(csv_dir, observable="X")
    repressilator.set_ShapeFile(csv_dir)

    # create physical effects
    physics = PhysicalEffects()
    physics.set_background(mean=0.01)
    physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    #physics.set_photobleaching(tau0=2.27, alpha=0.73)

    # create image
    create = EPIFMVisualizer(configs=repressilator, effects=physics)
    create.output_frames(num_div=16)



if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])

    test_repressilator(t0, t1)
