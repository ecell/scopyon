from bioimaging.epifm   import EPIFMConfigs, EPIFMVisualizer, EPIFMConfig
from bioimaging.effects import PhysicalEffects, PhysicalEffectsConfig
from bioimaging.image import convert_npy_to_8bit_image
import bioimaging.io as io

import logging
import os.path

import numpy
import glob


def test_epifm() :
    exposure_time = 0.100
    t0, t1 = 0.0, exposure_time * 2
    max_count = None # 20

    rndseed = 0
    rng = numpy.random.RandomState(rndseed)

    # create EPIFM imaging
    config = EPIFMConfig()
    config.set_shutter(start_time=t0, end_time=t1)
    config.set_light_source(source_type='LASER', wave_length=532, flux_density=40, angle=72)
    config.set_fluorophore(fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0, radius=20)
    config.set_dichroic_mirror('FF562-Di03-25x36')
    # config.set_magnification(magnification=100)
    # config.set_detector(detector='CMOS', image_size=(512, 512), pixel_length=6.5e-6, exposure_time=exposure_time, focal_point=(0.0, 0.5, 0.5), QE=0.73)
    # config.set_analog_to_digital_converter(bit=16, offset=100, fullwell=30000, fpn_type='column', fpn_count=2)
    config.set_magnification(magnification=241)
    config.set_detector(detector='EMCCD', image_size=(512, 512), pixel_length=16e-6, exposure_time=exposure_time, focal_point=(0.0, 0.5, 0.5), QE=0.92, readout_noise=100, emgain=300)
    config.set_analog_to_digital_converter(bit=16, offset=2000, fullwell=800000)

    epifm = EPIFMConfigs()
    epifm.initialize(config, rng=rng)

    # create physical effects
    config = PhysicalEffectsConfig()
    config.set_background(mean=0.01)
    config.set_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    config.set_photobleaching(tau0=2.27, alpha=0.73)

    physics = PhysicalEffects()
    physics.initialize(config)

    # read input data
    input_path = './scripts/data/inputs_epifm'
    dataset = io.read_spatiocyte(input_path, epifm.shutter_start_time, epifm.shutter_end_time, epifm.detector_exposure_time, 'S', max_count)

    # create image
    create = EPIFMVisualizer(configs=epifm, effects=physics)

    # bleaching
    new_dataset = create.apply_photophysics_effects(dataset, rng=rng)

    output_path = './scripts/data/outputs_tirf'
    create.output_frames(new_dataset, pathto=output_path, rng=rng)
    # create.output_frames(dataset, pathto=output_path, rng=rng)

    output_filenames = glob.glob(os.path.join(output_path, 'image_*.npy'))
    for filename in output_filenames:
        convert_npy_to_8bit_image(filename, cmin=1900, cmax=2500)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_epifm()

    # import sys
    # for filename in sys.argv[1: ]:
    #     convert_npy_to_8bit_image(filename, cmin=1900, cmax=2500)
