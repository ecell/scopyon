import sys
import os
import numpy

from lscm_handler   import LSCMConfigs, LSCMVisualizer
from effects_handler import PhysicalEffects

def test_lscm(t0, t1) :

    # create LSCM imaging
    lscm = LSCMConfigs()
    lscm.set_Shutter(start_time=t0, end_time=t1)
    lscm.set_LightSource(source_type='LASER', wave_length=532, flux=100e-6, radius=400e-9)
    #lscm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
    lscm.set_Fluorophore(fluorophore_type='EGFP')
    lscm.set_Pinhole(radius=28.8e-6)
    lscm.set_Magnification(Mag=60)

    # PMT : Photon-counting mode
    lscm.set_Detector(detector='PMT', mode="Photon-counting", image_size=(1024,1024), focal_point=(0.0,0.5,0.5), \
                    pixel_length=207.16e-9, scan_time=1.00, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
    lscm.set_ADConverter(bit=12, offset=0, fullwell=4096)

#       # PMT : Anlalog mode
#       lscm.set_Detector(detector='PMT', mode="Analog", image_size=(1024,1024), focal_point=(0.4,0.5,0.5), \
#                       pixel_length=207.16e-9, scan_time=1.15, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
#       lscm.set_ADConverter(bit=12, offset=0, fullwell=4096)

    # Output data
    lscm.set_OutputData(image_file_dir='./data_for_kakenhi/numpys/TestCell_00')

    # Input data
    csv_dir = '/home/masaki/wrk/spatiocyte/models/Kakenhi/data/TestCell_00'
    #csv_dir = '/home/masaki/bioimaging_4public/data_for_kakenhi/data_pb/...'
    lscm.set_InputFile(csv_dir, observable="R")
    lscm.set_ShapeFile(csv_dir)


    # create physical effects
    physics = PhysicalEffects()
    physics.set_background(mean=0.01)
    #physics.set_crosstalk(width=0.70)
    physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    physics.set_photobleaching(tau0=2.27, alpha=0.73)

    # create image
    create = LSCMVisualizer(configs=lscm, effects=physics)
    create.output_frames(num_div=16)



if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])

    test_lscm(t0, t1)
