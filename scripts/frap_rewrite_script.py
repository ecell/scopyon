import sys
import os
import numpy

from frap_handler   import FRAPConfigs, FRAPVisualizer
from effects_handler import PhysicalEffects

def test_frap(t0, t1) :

    # create FRAP imaging
    frap = FRAPConfigs()
    frap.set_LightSource(source_type='LASER', wave_length=532, imaging_flux=100e-6, bleaching_flux=100000e-3, \
                            bleaching_size=(30,30), bleaching_position=(512,512), bleaching_time=1.2, radius=400e-9)
    frap.set_Fluorophore(fluorophore_type='EGFP')
    frap.set_Pinhole(radius=28.8e-6)
    frap.set_Magnification(Mag=60)

    # PMT : Photon-counting mode
    frap.set_Detector(detector='PMT', mode="Photon-counting", image_size=(1024,1024), focal_point=(0.4,0.5,0.5), \
                    pixel_length=207.16e-9, scan_time=1.10, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
    frap.set_ADConverter(bit=12, offset=0, fullwell=4096)

#       # PMT : Anlalog mode
#       frap.set_Detector(detector='PMT', mode="Analog", image_size=(1024,1024), focal_point=(0.4,0.5,0.5), \
#                       pixel_length=207.16e-9, scan_time=1.15, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
#       frap.set_ADConverter(bit=12, offset=0, fullwell=4096)

#       # Output data
#       frap.set_OutputData(image_file_dir='./numpys_frap_001000A')
#       frap.set_OutputData(image_file_dir='./numpys_frap_019656A')

    # Input data
    #frap.reset_InputData('/home/masaki/wrk/spatiocyte/models/beads_3D_000100A', start=t0, end=t1, observable="A")
    #frap.reset_InputData('/home/masaki/wrk/spatiocyte/models/beads_3D_019656A', start=t0, end=t1, observable="A")
    frap.reset_InputData('/home/masaki/wrk/spatiocyte/models/beads_3D_080000A', start=t0, end=t1, observable="A")

    # create physical effects
    physics = PhysicalEffects()
    physics.set_background(mean=0)
    physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    physics.set_photobleaching(tau0=100e-6, alpha=0.73)
    #physics.set_photoactivation(turn_on_ratio=1000, activation_yield=1.0, frac_preactivation=0.00)
    #physics.set_photoblinking(t0_on=30e-6, a_on=0.58, t0_off=10e-6, a_off=0.48)

    # create image
    create = FRAPVisualizer(configs=frap, effects=physics)
    #create.rewrite_InputData(output_file_dir='./data_frap_000100A_bleach')
    #create.rewrite_InputData(output_file_dir='./data_frap_019656A_bleach')
    #create.rewrite_InputData(output_file_dir='./data_frap_019656A_blink')
    create.rewrite_InputData(output_file_dir='./data_frap_080000A_bleach')
    #create.output_frames(num_div=16)



if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])

    test_frap(t0, t1)
