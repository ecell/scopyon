import os.path
import pathlib
import math
import glob

import numpy

from bioimaging.io import read_spatiocyte
from bioimaging.config import Config
from bioimaging.epifm import EPIFMVisualizer
from bioimaging.image import convert_npy_to_8bit_image, convert_8bit, save_image_with_spots, show_with_spots, save_image
from bioimaging.spot_detection import spot_detection, blob_detection


def test_epifm() :
    exposure_time = 0.100
    # t0, t1 = 0.0, exposure_time * 100
    t0, t1 = 0.0, exposure_time * 2
    max_count = None # 20
    rndseed = 0
    cmin, cmax = 1900, 2500
    input_path = './scripts/data/inputs_epifm'
    output_path = './scripts/data/outputs_tirf'

    output_path_ = pathlib.Path(output_path)
    if not output_path_.exists():
        output_path_.mkdir()

    rng = numpy.random.RandomState(rndseed)

    ## read input data
    # dataset = read_spatiocyte(t0, t1, input_filename=os.path.join(input_path, 'pt-input.csv'), filenames=glob.glob(os.path.join(input_path, 'pt-000000??0.0.csv')), observable='S', max_count=max_count)
    dataset = read_spatiocyte(t0, t1, pathto=input_path, observable='S', max_count=max_count)

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

    ## create image
    sim = EPIFMVisualizer()
    sim.initialize(config, rng=rng)

    ## bleaching
    new_dataset = sim.apply_photophysics_effects(dataset, rng=rng)

    num_frames = math.ceil((t1 - t0) / exposure_time)
    print(num_frames)

    for i in range(num_frames):
        camera, true_data = sim.new_output_frame(new_dataset, i, rng=rng)
        # camera = numpy.load(output_path_.joinpath('image_{:07d}.npy'.format(i)))
        # true_data = numpy.load(output_path_.joinpath('true_{:07d}.npy'.format(i)))

        data = camera[: , : , 1]
        bytedata = convert_8bit(data, cmin, cmax)
        spots = spot_detection(data, min_sigma=2.0, max_sigma=4.0, num_sigma=20, threshold=10.0, overlap=0.5)

        numpy.save(str(output_path_.joinpath('image_{:07d}.npy'.format(i))), camera)
        numpy.save(str(output_path_.joinpath('true_{:07d}.npy'.format(i))), true_data)
        numpy.save(str(output_path_.joinpath('spot_{:07d}.npy'.format(i))), spots)
        # save_image(str(output_path_.joinpath('image_{:07d}.png'.format(i))), bytedata, low=0, high=255)
        save_image_with_spots(str(output_path_.joinpath('image_{:07d}.png'.format(i))), bytedata, spots, low=0, high=255)


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

    # intensity = []
    # for filename in pathlib.Path('./scripts/data/outputs_tirf').glob('spot_*.npy'):
    #     data = numpy.load(str(filename))
    #     intensity.extend(data.T[0])
    # intensity = numpy.asarray(intensity)

    # import matplotlib.pylab as plt
    # fold = 30
    # print(intensity.min(), intensity.max(), intensity.shape, intensity[intensity < intensity.min() * fold].size / intensity.size)
    # plt.hist(intensity, bins=30, range=(0.0, intensity.min() * fold))
    # plt.show()
