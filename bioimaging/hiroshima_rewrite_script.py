import sys
import os
import numpy

from hiroshima_handler   import HiroshimaConfigs, HiroshimaVisualizer
from effects_handler import PhysicalEffects

def test_hiroshima(t0, t1, cell) :

    # create Hiroshima imaging
    hiroshima = HiroshimaConfigs()
    hiroshima.set_Shutter(start_time=t0, end_time=t1)
    hiroshima.set_LightSource(source_type='LASER', wave_length=532, flux_density=50, angle=60)
    hiroshima.set_Fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0)
    hiroshima.set_DichroicMirror('FF562-Di03-25x36')
    hiroshima.set_Magnification(Mag=241)

    # Detector : EMCCD Camera
    hiroshima.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, exposure_time=0.050, \
                            focal_point=(0.92,0.5,0.5), QE=0.92, readout_noise=100, emgain=300)
    hiroshima.set_ADConverter(bit=16, offset=2000, fullwell=800000)

    # Output data
    #hiroshima.set_OutputData(image_file_dir='./numpys_')

    # Input data
    #csv_dir = '/home/masaki/wrk/spatiocyte/models/Hiroshima/data/SM_data/TestCells/Cell_00'
    csv_dir = '/home/masaki/wrk/spatiocyte/models/Hiroshima/data/SM_data/ModelCells/Dimer/config_B/Cell_%02d/2000pM' % (cell)
    hiroshima.set_InputFile(csv_dir, observable="R")
    hiroshima.set_ShapeFile(csv_dir)

    # create physical effects
    physics = PhysicalEffects()
    physics.set_background(mean=0.01)
    #physics.set_crosstalk(width=0.70)
    physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    physics.set_photobleaching(tau0=2.27, alpha=0.73)

    # create image
    create = HiroshimaVisualizer(configs=hiroshima, effects=physics)
    #output_file_dir = './data_hiroshima_for_calibration/data_pb/TestCell_00_tirfm'
    #output_file_dir = './data_hiroshima_for_calibration/data_pb/TestCell_00_epifm'
    output_file_dir = './data_hiroshima_for_calibration/data_pb/ModelCells/Dimer/Cell_%02d/2000pM' % (cell)
    create.rewrite_InputFile(output_file_dir=output_file_dir)
    #create.output_frames(num_div=16)



if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])
    cell = int(sys.argv[3])

    test_hiroshima(t0, t1, cell)
