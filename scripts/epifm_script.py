from bioimaging.epifm   import EPIFMConfigs, EPIFMVisualizer, EPIFMConfig
from bioimaging.effects import PhysicalEffects
import bioimaging.io as io

import logging
import os.path

import numpy
import glob


def test_epifm() :
    exposure_time = 0.050
    t0, t1 = 0.0, exposure_time * 2
    max_count = 20

    rndseed = 0
    rng = numpy.random.RandomState(rndseed)

    # create EPIFM imaging
    config = EPIFMConfig()
    config.set_shutter(start_time=t0, end_time=t1)
    config.set_light_source(source_type='LASER', wave_length=532, flux_density=20, angle=72)
    config.set_fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0)
    config.set_dichroic_mirror('FF562-Di03-25x36')
    config.set_magnification(magnification=364)
    config.set_detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, exposure_time=exposure_time,
                       focal_point=(0.0,0.5,0.5), QE=0.92, readout_noise=100, emgain=300)
    config.set_analog_to_digital_converter(bit=16, offset=2000, fullwell=800000)

    epifm = EPIFMConfigs()
    epifm.initialize(config, rng)

    # epifm.set_illumination_path(detector_focal_point=epifm.detector_focal_point, detector_focal_norm=(1.0, 0.0, 0.0))
    epifm.set_PSF_detector() # EPIFMVisualizer.__init__ -> epifm.set_optical_path() -> epifm.set_Detection_path()

    # create physical effects
    physics = PhysicalEffects()
    physics.set_background(mean=0.01)
    physics.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    physics.set_photobleaching(tau0=2.27, alpha=0.73)

    # read input data
    input_path = './scripts/data/inputs_epifm'
    dataset = io.read_spatiocyte(input_path, epifm.shutter_start_time, epifm.shutter_end_time, epifm.detector_exposure_time, 'S', max_count)

    # create image
    create = EPIFMVisualizer(configs=epifm, effects=physics)

    # bleaching
    new_data = create.rewrite_input_data(dataset, N_particles=max_count, rng=rng)

    output_path = './scripts/data/outputs_epifm'
    create.output_frames(rng, dataset, pathto=output_path)

    output_filenames = glob.glob('./scripts/data/outputs_epifm/image_*.npy')
    for filename in output_filenames:
        convert_npy_to_image(filename)

def convert_npy_to_image(filename):
    data = numpy.load(filename)
    data = data[: , : , 1]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.imshow(data, interpolation=None)
    output_filename = '{}.png'.format(os.path.splitext(filename)[0])
    plt.savefig(output_filename)
    # plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # import sys
    # t0 = float(sys.argv[1])
    # t1 = float(sys.argv[2])
    # test_epifm(t0, t1)

    test_epifm()

    # import sys
    # for filename in sys.argv[1: ]: convert_npy_to_image(filename)
