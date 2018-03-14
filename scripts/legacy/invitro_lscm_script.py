
import sys
import os

from pointscan_confm_handler import PointScanConfocalConfigs, PointScanConfocalVisualizer
from effects_handler import PhysicalEffects

def confm_test(t0, t1) :

    # create Point-scanning Confocal Microscopy
    confm = PointScanConfocalConfigs()

    confm.set_LightSource(source_type='LASER', wave_length=488, flux=100e-6, radius=200e-9)
    confm.set_Fluorophore(fluorophore_type='EGFP')
    confm.set_Pinhole(radius=28.8e-6)
    confm.set_Magnification(Mag=60)

    # PMT : Analog-mode
    confm.set_Detector(detector='PMT', mode="Analog", image_size=(1024,1024), focal_point=(0.3,0.5,0.5), \
                    pixel_length=207.16e-9, scan_time=1.15, QE=0.3, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
    confm.set_ADConverter(bit=16, offset=0, fullwell=65536)
    #confm.set_OutputData(image_file_dir='./images_erk102_invitro_01')
    #confm.set_OutputData(image_file_dir='./images_erk102_invitro_1000nM')
    confm.set_OutputData(image_file_dir='./images_EGF_model_3_coord_erk_0_0100')

    # Input data : EGF model file
    #confm.set_InputData('/home/masaki/ecell3/latest/data/csv/invitro_01', start=t0, end=t1, observable="A")
    #confm.set_InputData('/home/masaki/ecell3/latest/data/csv/erk102_invitro_1000nM', start=t0, end=t1, observable="A")
    confm.set_InputData('/home/masaki/ecell3/latest/data/csv/EGF_model_3_coord_erk_0_0100', start=t0, end=t1, observable="ERK")

    # create physical effects
    physics = PhysicalEffects()
    physics.set_Conversion(ratio=1e-6)
    #physics.set_Background(mean=5)
    #physics.set_DetectorCrosstalk(width=1.00)

    # create image and movie
    create = PointScanConfocalVisualizer(configs=confm, effects=physics)
    create.output_frames(num_div=16)



if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])

    confm_test(t0, t1)
