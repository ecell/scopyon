import sys
import os
import numpy

from epifm_handler   import EPIFMConfigs, EPIFMVisualizer
from effects_handler import PhysicalEffects

def test_oscillator(t0, t1) :

    # create TIRFM imaging
    oscillator = EPIFMConfigs()
    oscillator.set_Shutter(start_time=t0, end_time=t1)
    #oscillator.set_Shutter(start_time=t0, end_time=t1, time_open=0.150, time_lapse=20)
    oscillator.set_LightSource(source_type='LASER', wave_length=532, flux_density=20, angle=72)
    oscillator.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0)
    oscillator.set_DichroicMirror('FF562-Di03-25x36')
    oscillator.set_Magnification(Mag=234)

    # Detector : EMCCD Camera
    oscillator.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, exposure_time=0.20, \
                            focal_point=(0.0,0.5,0.5), QE=0.92, readout_noise=100, emgain=300)
    oscillator.set_ADConverter(bit=16, offset=2000, fullwell=800000)

    # Output data
    oscillator.set_OutputData(image_file_dir='./data_for_nikon/numpys/oscillator00')

    # Input data
    csv_dir = '/home/masaki/wrk/spatiocyte/models/Nikon/data/oscillator00'
    oscillator.set_InputFile(csv_dir, observable="Min")
    oscillator.set_ShapeFile(csv_dir)

    # create physical effects
    physics = PhysicalEffects()
    physics.set_background(mean=0.01)
    physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    #physics.set_photobleaching(tau0=2.27, alpha=0.73)

    # create image
    create = EPIFMVisualizer(configs=oscillator, effects=physics)
    create.output_frames(num_div=16)



if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])

    test_oscillator(t0, t1)
