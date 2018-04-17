import os.path
import glob

import numpy

from bioimaging.io import read_spatiocyte
from bioimaging.config import Config
from bioimaging.epifm import EPIFMVisualizer
from bioimaging.image import convert_npy_to_8bit_image, convert_8bit, save_image_with_spots, show_with_spots
from bioimaging.spot_detection import spot_detection, blob_detection


def test_epifm() :
    exposure_time = 0.100
    t0, t1 = 0.0, exposure_time * 2
    max_count = None # 20
    rndseed = 0
    input_path = './scripts/data/inputs_epifm'
    output_path = './scripts/data/outputs_tirf'
    cmin, cmax = 1900, 2500

    rng = numpy.random.RandomState(rndseed)

    ## read input data
    dataset = read_spatiocyte(t0, t1, input_path, observable='S', max_count=max_count)

    config = Config()

    ## set epifm configurations
    config.set_epifm_shutter(start_time=t0, end_time=t1)
    config.set_epifm_light_source(source_type='LASER', wave_length=532, flux_density=40, angle=72)
    config.set_epifm_fluorophore(
        fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0, radius=20)
    config.set_epifm_dichroic_mirror('FF562-Di03-25x36')
    # config.set_epifm_magnification(magnification=100)
    # config.set_epifm_detector(
    #     detector='CMOS', image_size=(512, 512), pixel_length=6.5e-6, exposure_time=exposure_time,
    #     focal_point=(0.0, 0.5, 0.5), QE=0.73)
    # config.set_epifm_analog_to_digital_converter(
    #     bit=16, offset=100, fullwell=30000, fpn_type='column', fpn_count=2)
    config.set_epifm_magnification(magnification=241)
    config.set_epifm_detector(
        detector='EMCCD', image_size=(512, 512), pixel_length=16e-6, exposure_time=exposure_time,
        focal_point=(0.0, 0.5, 0.5), QE=0.92, readout_noise=100, emgain=300)
    config.set_epifm_analog_to_digital_converter(bit=16, offset=2000, fullwell=800000)

    ## set effects configurations
    config.set_effects_background(mean=0.01)
    config.set_effects_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    config.set_effects_photobleaching(tau0=2.27, alpha=0.73)

    # ## create image
    # sim = EPIFMVisualizer()
    # sim.initialize(config, rng=rng)

    # ## bleaching
    # new_dataset = sim.apply_photophysics_effects(dataset, rng=rng)

    # sim.output_frames(new_dataset, pathto=output_path, rng=rng)
    # # sim.output_frames(dataset, pathto=output_path, rng=rng)

    output_filenames = glob.glob(os.path.join(output_path, 'image_*.npy'))

    for filename in output_filenames:
        # convert_npy_to_8bit_image(filename, cmin=cmin, cmax=cmax)

        data = numpy.load(filename)
        data = data[: , : , 1]
        bytedata = convert_8bit(data, cmin, cmax, 0, 255)

        ## spot-detection
        spots = spot_detection(data, min_sigma=2.0, max_sigma=4.0, num_sigma=20, threshold=10.0, overlap=0.5)

        ## draw spots
        save_image_with_spots('{}.png'.format(os.path.splitext(filename)[0]), bytedata, spots, low=0, high=255)

        # blobs = blob_detection(data, min_sigma=2.0, max_sigma=4.0, num_sigma=20, threshold=10.0, overlap=0.5)
        # spots = spot_detection(data, blobs=blobs, opt=1)
        # show_with_spots(bytedata, spots, blobs, low=0, high=255)


if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(fmt)
    log = logging.getLogger('bioimaging')
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    test_epifm()

    # import sys
    # for filename in sys.argv[1: ]:
    #     convert_npy_to_8bit_image(filename, cmin=1900, cmax=2500)
