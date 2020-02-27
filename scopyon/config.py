from .configbase import _Config


class Config(_Config):

    def __init__(self, *args, **kwargs):
        _Config.__init__(self, *args, **kwargs)

    def set_epifm_shutter(self, start_time=None, end_time=None, time_open=None, time_lapse=None, switch=True):
        self.update('shutter_switch', switch)
        self.update('shutter_start_time', start_time)
        self.update('shutter_end_time', end_time)
        self.update('shutter_time_open', time_open)
        self.update('shutter_time_lapse', time_lapse)

    def set_epifm_light_source(
            self, source_type=None, wave_length=None, flux_density=None, radius=None, angle=None, switch=True):
        self.update('source_switch', switch)
        self.update('source_type', source_type)
        self.update('source_wavelength', wave_length)
        self.update('source_flux_density', flux_density)
        self.update('source_radius', radius)
        self.update('source_angle', angle)

    def set_epifm_fluorophore(
            self, fluorophore_type=None, wave_length=None, normalization=None, radius=None, width=None, cutoff=None,
            file_name_format=None):
        self.update('fluorophore_type', fluorophore_type)
        self.update('fluorophore_radius', radius)
        self.update('psf_wavelength', wave_length)
        self.update('psf_normalization', normalization)
        self.update('psf_width', width)
        self.update('psf_cutoff', cutoff)
        self.update('psf_file_name_format', file_name_format)

    def set_epifm_dichroic_mirror(self, dm=None, switch=True):
        self.update('dichroic_switch', switch)
        self.update('dichroic_mirror', dm)

    def set_epifm_magnification(self, magnification=None):
        self.update('image_magnification', magnification)

    def set_epifm_detector(
            self, detector=None, image_size=None, pixel_length=None, exposure_time=None, focal_point=None,
            QE=None, readout_noise=None, dark_count=None, emgain=None, switch=True):
        self.update('detector_switch', switch)
        self.update('detector_type', detector)
        self.update('detector_image_size', image_size)
        self.update('detector_pixel_length', pixel_length)
        self.update('detector_exposure_time', exposure_time)
        self.update('detector_focal_point', focal_point)
        self.update('detector_qeff', QE)
        self.update('detector_readout_noise', readout_noise)
        self.update('detector_dark_count', dark_count)
        self.update('detector_emgain', emgain)

    def set_epifm_analog_to_digital_converter(self, bit=None, offset=None, fullwell=None, fpn_type=None, fpn_count=None):
        self.update('ADConverter_bit', bit)
        self.update('ADConverter_offset', offset)
        self.update('ADConverter_fullwell', fullwell)
        self.update('ADConverter_fpn_type', fpn_type)
        self.update('ADConverter_fpn_count', fpn_count)

    def set_epifm_illumination_path(self, focal_point, focal_norm):
        self.update('detector_focal_point', focal_point)
        self.update('detector_focal_norm', focal_norm)

    def set_epifm_excitation_filter(self, excitation=None, switch=True):
        self.update('excitation_switch', switch)
        self.update('excitation_filter', excitation)

    def set_epifm_emission_filter(self, emission=None, switch=True):
        self.update('emission_switch', switch)
        self.update('emission_filter', emission)

    def set_effects_background(self, mean=None):
        self.update('background_switch', True)
        self.update('background_mean', mean)

    def set_effects_fluorescence(self, quantum_yield=None, abs_coefficient=None):
        self.update('quantum_yield', quantum_yield)
        self.update('abs_coefficient', abs_coefficient)

    def set_effects_photobleaching(self, tau0=None, alpha=None):
        self.update('photobleaching_switch', True)
        self.update('photobleaching_tau0', tau0)
        self.update('photobleaching_alpha', alpha)

    def set_effects_photoactivation(self, turn_on_ratio=None, activation_yield=None, frac_preactivation=None):
        self.update('photoactivation_switch', True)
        self.update('photoactivation_turn_on_ratio', turn_on_ratio)
        self.update('photoactivation_activation_yield', activation_yield)
        self.update('photoactivation_frac_preactivation', frac_preactivation)

    def set_effects_photoblinking(self, t0_on=None, a_on=None, t0_off=None, a_off=None):
        self.update('photoblinking_switch', True)
        self.update('photoblinking_t0_on', t0_on)
        self.update('photoblinking_a_on', a_on)
        self.update('photoblinking_t0_off', t0_off)
        self.update('photoblinking_a_off', a_off)

