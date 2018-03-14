import sys
import os
import numpy

from lscm_handler   import LSCMConfigs, LSCMVisualizer
from effects_handler import PhysicalEffects

def test_lscm(t0, t1) :

    # create LSCM imaging
    lscm = LSCMConfigs()
    lscm.set_LightSource(source_type='LASER', wave_length=532, flux=100e-6, radius=400e-9)
    #lscm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
    lscm.set_Fluorophore(fluorophore_type='EGFP')
    lscm.set_Pinhole(radius=28.8e-6)
    lscm.set_Magnification(Mag=60)

    # PMT : Photon-counting mode
    lscm.set_Detector(detector='PMT', mode="Photon-counting", image_size=(1024,1024), focal_point=(0.4,0.5,0.5), \
                    pixel_length=207.16e-9, scan_time=1.10, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
    lscm.set_ADConverter(bit=12, offset=0, fullwell=4096)

#       # PMT : Anlalog mode
#       lscm.set_Detector(detector='PMT', mode="Analog", image_size=(1024,1024), focal_point=(0.4,0.5,0.5), \
#                       pixel_length=207.16e-9, scan_time=1.15, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
#       lscm.set_ADConverter(bit=12, offset=0, fullwell=4096)

#       # Output data
#       lscm.set_OutputData(image_file_dir='./numpys_lscm_001000A')
#       lscm.set_OutputData(image_file_dir='./numpys_lscm_019656A')

    # Input data
    lscm.reset_InputData('/home/masaki/wrk/spatiocyte/models/.....', start=t0, end=t1, observable="A")

    # create physical effects
    physics = PhysicalEffects()
    physics.set_background(mean=0)
    physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    physics.set_photobleaching(tau0=100e-6, alpha=0.73)
    #physics.set_photoactivation(turn_on_ratio=1000, activation_yield=1.0, frac_preactivation=0.00)
    #physics.set_photoblinking(t0_on=30e-6, a_on=0.58, t0_off=10e-6, a_off=0.48)

    # create image
    create = LSCMVisualizer(configs=lscm, effects=physics)
    create.rewrite_InputData(output_file_dir='./data_lscm_for_kakenhi/data_pb/.....')
    #create.output_frames(num_div=16)



if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])

    test_lscm(t0, t1)
