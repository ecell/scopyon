import numpy

from bioimaging.io import read_spatiocyte
from bioimaging.config import Config
from bioimaging.epifm import EPIFMSimulator
from bioimaging.image import convert_8bit, save_image_with_spots, show_with_spots, save_image
from bioimaging.spot_detection import spot_detection, blob_detection


def test_tirf() :
    exposure_time = 0.100
    # t0, t1 = 0.0, exposure_time * 100
    t0, t1 = 0.0, exposure_time * 2
    max_count = None # 20
    rndseed = 0
    cmin, cmax = 1900, 2500
    input_path = './scripts/data/inputs_epifm'
    output_path = './scripts/data/outputs_tirf'

    rng = numpy.random.RandomState(rndseed)

    ## read input data
    # import os.path
    # import glob
    # (input_data, lengths) = read_spatiocyte(t0, t1, input_filename=os.path.join(input_path, 'pt-input.csv'), filenames=glob.glob(os.path.join(input_path, 'pt-000000??0.0.csv')), observable='S', max_count=max_count)
    (input_data, lengths) = read_spatiocyte(t0, t1, pathto=input_path, observable='S', max_count=max_count)

    config = Config()

    ## set epifm configurations
    config.set_epifm_shutter(start_time=t0, end_time=t1)
    config.set_epifm_light_source(source_type='LASER', wave_length=532, flux_density=40, angle=72)
    config.set_epifm_fluorophore(
        fluorophore_type='Tetramethylrhodamine(TRITC)', normalization=1.0, radius=20)
    config.set_epifm_dichroic_mirror('FF562-Di03-25x36')
    focal_point = numpy.array([lengths[0] * 0.0, lengths[1] * 0.5, lengths[2] * 0.5])
    # config.set_epifm_magnification(magnification=100)
    # config.set_epifm_detector(
    #     detector='CMOS', image_size=(512, 512), pixel_length=6.5e-6, exposure_time=exposure_time,
    #     focal_point=focal_point, QE=0.73)
    # config.set_epifm_analog_to_digital_converter(
    #     bit=16, offset=100, fullwell=30000, fpn_type='column', fpn_count=2)
    config.set_epifm_magnification(magnification=241)
    config.set_epifm_detector(
        detector='EMCCD', image_size=(512, 512), pixel_length=16e-6, exposure_time=exposure_time,
        focal_point=focal_point, QE=0.92, readout_noise=100, emgain=300)
    config.set_epifm_analog_to_digital_converter(bit=16, offset=2000, fullwell=800000)

    ## set effects configurations
    config.set_effects_background(mean=0.01)
    config.set_effects_fluorescence(quantum_yield=0.61, abs_coefficient=83400)
    config.set_effects_photobleaching(tau0=2.27, alpha=0.73)

    # config.write('tirf_script.ini')

    ## create image
    sim = EPIFMSimulator(config, rng=rng)

    ## bleaching
    new_input_data = sim.apply_photophysics_effects(input_data, rng=rng)

    ## output frame data
    sim.output_frames(new_input_data, pathto=output_path, cmin=cmin, cmax=cmax, rng=rng)


if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(fmt)
    log = logging.getLogger('bioimaging')
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    test_tirf()
