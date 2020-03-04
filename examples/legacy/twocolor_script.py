import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

from scopyon.image import convert_8bit
from scopyon.io import read_spatiocyte
from scopyon.config import Config
from scopyon.epifm import EPIFMSimulator


def test_tirf() :
    exposure_time = 0.100
    # t0, t1 = 0.0, exposure_time * 100
    t0, t1 = 0.0, exposure_time * 2
    max_count = None # 20
    rndseed = 0
    cmin, cmax = 1900, 2500
    input_path = './examples/data/inputs_epifm'
    output_path = './examples/data/outputs_tirf'

    rng = numpy.random.RandomState(rndseed)

    ## read input data
    (input_data1, lengths) = read_spatiocyte(t0, t1, pathto=input_path, observable='S1', max_count=max_count)
    (input_data2, _) = read_spatiocyte(t0, t1, pathto=input_path, observable='S2', max_count=max_count)

    print(lengths)

    config = Config()

    ## set epifm configurations
    config.set_epifm_shutter(start_time=t0, end_time=t1)
    config.set_epifm_light_source(source_type='LASER', wave_length=532e-9, flux_density=40, angle=72)
    config.set_epifm_fluorophore(
        fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0, radius=20e-9)
    config.set_epifm_dichroic_mirror('FF562-Di03-25x36')
    focal_point = numpy.array([lengths[0] * 0.0, lengths[1] * 0.5, lengths[2] * 0.5])
    config.set_epifm_magnification(magnification=241)
    config.set_epifm_detector(
        detector='EMCCD', image_size=(512, 512), pixel_length=16e-6, exposure_time=exposure_time,
        focal_point=focal_point, QE=0.92, readout_noise=100, emgain=300)
    config.set_epifm_analog_to_digital_converter(bit=16, offset=2000, fullwell=800000)

    ## set effects configurations
    config.set_effects_background(mean=0.01)
    config.set_effects_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    config.set_effects_photobleaching(tau0=2.27, alpha=0.73)

    ## create image
    sim = EPIFMSimulator(config, rng=rng)

    ## bleaching
    input_data1 = sim.apply_photophysics_effects(input_data1, rng=rng)
    input_data2 = sim.apply_photophysics_effects(input_data2, rng=rng)

    low, high = 0, 255
    Nw_pixel, Nh_pixel = sim.configs.detector_image_size  # pixels
    for frame_index in range(sim.num_frames()):
        bytedata = numpy.zeros((Nw_pixel, Nh_pixel, 3), numpy.uint8)

        camera, _ = sim.output_frame(input_data1, frame_index, rng=rng)
        bytedata[:, :, 0] = convert_8bit(camera[: , : , 1], cmin, cmax, low, high)
        camera, _ = sim.output_frame(input_data2, frame_index, rng=rng)
        bytedata[:, :, 1] = convert_8bit(camera[: , : , 1], cmin, cmax, low, high)

        image_file_name = os.path.join(output_path, 'twocolor_%07d.png' % (frame_index))
        plt.imsave(image_file_name, bytedata, vmin=low, vmax=high, dpi=100)
        plt.clf()


if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(fmt)
    log = logging.getLogger('scopyon')
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    test_tirf()
