
import sys
import os

from pointscan_confm_handler import PointScanConfocalConfigs, PointScanConfocalVisualizer
from effects_handler import PhysicalEffects

def test_confm(t0, t1, beam) :

    # create Point-scanning Confocal Microscopy
    confm = PointScanConfocalConfigs()

    confm.set_LightSource(source_type='LASER', wave_length=488, flux=beam*1e-6, radius=200e-9)
    #confm.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)')
    confm.set_Fluorophore(fluorophore_type='EGFP')
    confm.set_Pinhole(radius=28.8e-6)
    confm.set_Magnification(Mag=60)

#       # PMT : Photon-counting mode
#       confm.set_Detector(detector='PMT', mode="Photon-counting", image_size=(1024,1024), focal_point=(0.3,0.5,0.5), \
#                       pixel_length=210e-9, scan_time=1.00, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
#       confm.set_ADConverter(bit=12, offset=0, fullwell=4096)

    # PMT : Anlalog mode
    confm.set_Detector(detector='PMT', mode="Analog", image_size=(1024,1024), focal_point=(0.3,0.5,0.5), \
                    pixel_length=210e-9, scan_time=1.00, gain=1e+6, dyn_stages=11, pair_pulses=18e-9)
    confm.set_ADConverter(bit=12, offset=0, fullwell=4096)

    ### Output data
    #confm.set_OutputData(image_file_dir='./images_tmr_lscm')
    confm.set_OutputData(image_file_dir='./images_lscm_test_%03duW' % (beam))

    ### Input data
    #confm.set_InputData('./data/tmr_lscm', start=t0, end=t1, observable="A")
    confm.set_InputData('/home/masaki/ecell3/latest/data/csv/beads_07', start=t0, end=t1, observable="A")

    # create physical effects
    physics = PhysicalEffects()
    physics.set_Conversion(ratio=1e-6)
    #physics.set_Background(mean=3)

    # create image and movie
    create = PointScanConfocalVisualizer(configs=confm, effects=physics)
    create.output_frames(num_div=16)



if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])
    beam = float(sys.argv[3])

    test_confm(t0, t1, beam)
