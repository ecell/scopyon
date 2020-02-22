from logging import getLogger
_log = getLogger(__name__)

import numpy
import collections.abc
import warnings

from scipy.special import j0, i1e
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

import scopyon.epifm
import scopyon.config
from scopyon import io
from scopyon import constants


def levy_probability_function(t, t0, a):
    return (a / t0) * numpy.power(t0 / t, 1 + a)
    # return numpy.power(t0 / t, 1 + a)

class PhysicalEffects:
    '''
    Physical effects setting class

        Fluorescence
        Photo-bleaching
        Photo-blinking
    '''

    def __init__(self, config):
        self.initialize(config)

    def initialize(self, config):
        self.fluorescence_bleach = []
        self.fluorescence_budget = []
        self.fluorescence_state  = []

        self.set_background(**config.background)
        self.set_crosstalk(**config.crosstalk)
        self.set_fluorescence(**config.fluorescence)
        self.set_photobleaching(**config.photo_bleaching)
        self.set_photoactivation(**config.photo_activation)
        self.set_photoblinking(**config.photo_blinking)

    def _set_data(self, key, val):
        if val != None:
            setattr(self, key, val)

    def set_background(self, mean=None, switch=True):
        self.background_switch = switch
        self.background_mean = mean
        _log.info('--- Background: ')
        _log.info('    Mean = {} photons'.format(self.background_mean))

    def set_crosstalk(self, width=None, switch=True):
        self.crosstalk_switch = switch
        self.crosstalk_width = width
        _log.info('--- Photoelectron Crosstalk: ')
        _log.info('    Width = {} pixels'.format(self.crosstalk_width))

    def set_fluorescence(self, quantum_yield=None, abs_coefficient=None):
        self.quantum_yield = quantum_yield
        self.abs_coefficient = abs_coefficient
        _log.info('--- Fluorescence: ')
        _log.info('    Quantum Yield =  {}'.format(self.quantum_yield))
        _log.info('    Abs. Coefficient =  {} 1/(M cm)'.format(self.abs_coefficient))
        _log.info('    Abs. Cross-section =  {} cm^2'.format((numpy.log(10) * self.abs_coefficient * 0.1 / constants.N_A) * 1e+4))

    def set_photobleaching(self, tau0=None, alpha=None, switch=True):
        self.photobleaching_switch = switch
        self.photobleaching_tau0 = tau0
        self.photobleaching_alpha = alpha
        _log.info('--- Photobleaching: ')
        _log.info('    Photobleaching tau0  =  {}'.format(self.photobleaching_tau0))
        _log.info('    Photobleaching alpha =  {}'.format(self.photobleaching_alpha))

    def set_photoactivation(self, turn_on_ratio=None, activation_yield=None, frac_preactivation=None, switch=True):
        self.photoactivation_switch = switch
        self.photoactivation_turn_on_ratio = turn_on_ratio
        self.photoactivation_activation_yield = activation_yield
        self.photoactivation_frac_preactivation = frac_preactivation
        _log.info('--- Photoactivation: ')
        _log.info('    Turn-on Ratio  =  {}'.format(self.photoactivation_turn_on_ratio))
        _log.info('    Effective Ratio  =  {}'.format(self.photoactivation_activation_yield * self.photoactivation_turn_on_ratio / (1 + self.photoactivation_frac_preactivation * self.photoactivation_turn_on_ratio)))
        _log.info('    Reaction Yield =  {}'.format(self.photoactivation_activation_yield))
        _log.info('    Fraction of Preactivation =  {}'.format(self.photoactivation_frac_preactivation))

    def set_photoblinking(self, t0_on=None, a_on=None, t0_off=None, a_off=None, switch=True):
        self.photoblinking_switch = switch
        self.photoblinking_t0_on = t0_on
        self.photoblinking_a_on = a_on
        self.photoblinking_t0_off = t0_off
        self.photoblinking_a_off = a_off
        _log.info('--- Photo-blinking: ')
        _log.info('    (ON)  t0 =  {} sec'.format(self.photoblinking_t0_on))
        _log.info('    (ON)  a  =  {}'.format(self.photoblinking_a_on))
        _log.info('    (OFF) t0 =  {} sec'.format(self.photoblinking_t0_off))
        _log.info('    (OFF) a  =  {}'.format(self.photoblinking_a_off))

    def get_prob_bleach(self, tau, dt):
        # set the photobleaching-time
        tau0  = self.photobleaching_tau0
        alpha = self.photobleaching_alpha

        # set probability
        prob = self.prob_levy(tau, tau0, alpha)
        norm = prob.sum()
        p_bleach = prob/norm

        return p_bleach

    def get_prob_blink(self, tau_on, tau_off, dt):
        # time scale
        t0_on  = self.photoblinking_t0_on
        a_on   = self.photoblinking_a_on
        t0_off = self.photoblinking_t0_off
        a_off  = self.photoblinking_a_off

        # ON-state
        prob_on = self.prob_levy(tau_on, t0_on, a_on)
        norm_on = prob_on.sum()
        p_blink_on = prob_on/norm_on

        # OFF-state
        prob_off = self.prob_levy(tau_off, t0_off, a_off)
        norm_off = prob_off.sum()
        p_blink_off = prob_off/norm_off

        return p_blink_on, p_blink_off

    def get_photobleaching_property(self, dt, n_emit0, rng):
        # set the photobleaching-time
        tau0  = self.photobleaching_tau0

        # set photon budget
        photon0 = (tau0/dt)*n_emit0

        tau_bleach = numpy.array([j*dt + tau0 for j in range(int(1e+7))])
        p_bleach = self.get_prob_bleach(tau_bleach, dt)

        # get photon budget and photobleaching-time
        tau = rng.choice(tau_bleach, 1, p=p_bleach)
        budget = (tau/tau0)*photon0

        return tau, budget

    def get_photophysics_for_epifm(self, time_array, N_emit0, N_part, rng=None):
        """
        Args:
            N_emit0 (float): The number of photons emitted per unit time.
        """
        if self.photobleaching_switch is False:
            raise RuntimeError('Not supported')
        elif rng is None:
            raise RuntimeError('A random number generator is required.')

        # set the photobleaching-time
        tau0  = self.photobleaching_tau0
        alpha = self.photobleaching_alpha
        prob_func = functools.partial(levy_probability_function, t0=tau0, a=alpha)

        # set photon budget
        # photon0 = tau0 * N_emit0

        dt = tau0 * 1e-3
        tau_bleach = numpy.arange(tau0, 50001 * tau0, dt)
        prob = prob_func(tau_bleach)
        norm = prob.sum()
        p_bleach = prob / norm

        # get photobleaching-time
        tau = rng.choice(tau_bleach, N_part, p=p_bleach)

        # sequences
        budget = numpy.zeros(N_part)
        state = numpy.zeros((N_part, len(time_array)))
        for i in range(N_part):
            # get photon budget
            budget[i] = tau[i] * N_emit0

            # bleaching time
            Ni = numpy.searchsorted(time_array, tau[i])
            state[i][0: Ni] = 1.0

        return state, budget

    def set_photophysics_4palm(self, start, end, dt, f, F, N_part, rng):
        ##### PALM Configuration
        NNN = int(1e+7)

        # set the photobleaching-time
        tau0  = self.photobleaching_tau0
        alpha = self.photobleaching_alpha

        # set photon budget
        # photon0 = (tau0/dt)*N_emit0
        photon0 = (tau0/dt)*1.0

        tau_bleach = numpy.array([j*dt+tau0 for j in range(NNN)])
        prob = self.prob_levy(tau_bleach, tau0, alpha)
        norm = prob.sum()
        p_bleach = prob/norm

        # set photoblinking (ON/OFF) probability density function (PDF)
        if (self.photoblinking_switch == True):

            # time scale
            t0_on  = self.photoblinking_t0_on
            a_on   = self.photoblinking_a_on
            t0_off = self.photoblinking_t0_off
            a_off  = self.photoblinking_a_off

            tau_on  = numpy.array([j*dt+t0_on  for j in range(NNN)])
            tau_off = numpy.array([j*dt+t0_off for j in range(NNN)])

            # ON-state
            prob_on = self.prob_levy(tau_on, t0_on, a_on)
            norm_on = prob_on.sum()
            p_on = prob_on/norm_on

            # OFF-state
            prob_off = self.prob_levy(tau_off, t0_off, a_off)
            norm_off = prob_off.sum()
            p_off = prob_off/norm_off

        # sequences
        budget = []
        state_act = []

        # get random number
        # numpy.random.seed()

        # get photon budget and photobleaching-time
        tau = rng.choice(tau_bleach, N_part, p=p_bleach)

        for i in range(N_part):

            # get photon budget and photobleaching-time
            photons = (tau[i]/tau0)*photon0

            N = int((end - start)/dt)

            time  = numpy.array([j*dt + start for j in range(N)])
            state = numpy.zeros(shape=(N))

            #### Photoactivation: overall activation yeild
            p = self.photoactivation_activation_yield
            # q = self.photoactivation_frac_preactivation

            r = rng.uniform(0, 1, N/F*f)
            s = (r < p).astype('int')

            index = numpy.abs(s - 1).argmin()
            t0 = (index/f*F + numpy.remainder(index, f))*dt + start

            t1 = t0 + tau[i]

            N0 = (numpy.abs(time - t0)).argmin()
            N1 = (numpy.abs(time - t1)).argmin()
            state[N0:N1] = numpy.ones(shape=(N1-N0))

            if (self.photoblinking_switch == True):

                # ON-state
                t_on  = rng.choice(tau_on,  100, p=p_on)
                # OFF-state
                t_off = rng.choice(tau_off, 100, p=p_off)

                k = (numpy.abs(numpy.cumsum(t_on) - tau[i])).argmin()

                # merge t_on/off arrays
                t_blink = numpy.array([t_on[0:k], t_off[0:k]])
                t = numpy.reshape(t_blink, 2*k, order='F')

                i_on  = N0

                for j in range(k):

                    f_on  = i_on  + int(t[j]/dt)
                    i_off = f_on
                    f_off = i_off + int(t[j+1]/dt)

                    state[i_on:f_on]   = 1
                    state[i_off:f_off] = 0

                    i_on = f_off

            # set fluorescence state (without photoblinking)
            budget.append(photons)
            state_act.append(state)

        self.fluorescence_bleach = numpy.array(tau)
        self.fluorescence_budget = numpy.array(budget)
        self.fluorescence_state  = numpy.array(state_act)

    def set_photophysics_4lscm(self, start, end, dt, N_part, rng):
        ##### LSCM Configuration

        NNN = int(1e+7)

        # set the photobleaching-time
        tau0  = self.photobleaching_tau0
        alpha = self.photobleaching_alpha

        # set photon budget
        photon0 = (tau0/dt)*1.0

        tau_bleach = numpy.array([j*dt+tau0 for j in range(NNN)])
        prob = self.prob_levy(tau_bleach, tau0, alpha)
        norm = prob.sum()
        p_bleach = prob/norm

        # set photoblinking (ON/OFF) probability density function (PDF)
        if (self.photoblinking_switch == True):

            # time scale
            t0_on  = self.photoblinking_t0_on
            a_on   = self.photoblinking_a_on
            t0_off = self.photoblinking_t0_off
            a_off  = self.photoblinking_a_off

            tau_on  = numpy.array([j*dt+t0_on  for j in range(NNN)])
            tau_off = numpy.array([j*dt+t0_off for j in range(NNN)])

            # ON-state
            prob_on = self.prob_levy(tau_on, t0_on, a_on)
            norm_on = prob_on.sum()
            p_on = prob_on/norm_on

            # OFF-state
            prob_off = self.prob_levy(tau_off, t0_off, a_off)
            norm_off = prob_off.sum()
            p_off = prob_off/norm_off

        # sequences
        budget = []
        state_act = []

        # get random number
        # numpy.random.seed()

        # get photon budget and photobleaching-time
        tau = rng.choice(tau_bleach, N_part, p=p_bleach)

        for i in range(N_part):

            # get photon budget and photobleaching-time
            photons = (tau[i]/tau0)*photon0

            N = 10000 #int((end - start)/dt)

            time  = numpy.array([j*dt + start for j in range(N)])
            state = numpy.zeros(shape=(N))

            t0 = start
            t1 = t0 + tau[i]

            N0 = (numpy.abs(time - t0)).argmin()
            N1 = (numpy.abs(time - t1)).argmin()
            state[N0:N1] = numpy.ones(shape=(N1-N0))

            if (self.photoblinking_switch == True):

                # ON-state
                t_on  = rng.choice(tau_on,  100, p=p_on)
                # OFF-state
                t_off = rng.choice(tau_off, 100, p=p_off)

                k = (numpy.abs(numpy.cumsum(t_on) - tau[i])).argmin()

                # merge t_on/off arrays
                t_blink = numpy.array([t_on[0:k], t_off[0:k]])
                t = numpy.reshape(t_blink, 2*k, order='F')

                i_on  = N0

                for j in range(k):

                    f_on  = i_on  + int(t[j]/dt)
                    i_off = f_on
                    f_off = i_off + int(t[j+1]/dt)

                    state[i_on:f_on]   = 1
                    state[i_off:f_off] = 0

                    i_on = f_off

            # set photon budget and fluorescence state
            budget.append(photons)
            state_act.append(state)

        self.fluorescence_bleach = numpy.array(tau)
        self.fluorescence_budget = numpy.array(budget)
        self.fluorescence_state  = numpy.array(state_act)

class _EPIFMConfigs:

    def __init__(self, config, rng=None):
        self.initialize(config, rng=rng)

    def initialize(self, config, rng=None):
        """Initialize based on the given config.

        Args:
            config (Configuration): A config object.
            rng (numpy.RandomState, optional): A random number generator
                for initializing this class. Defaults to `None`.

        """
        if rng is None:
            warnings.warn('A random number generator [rng] is not given.')
            rng = numpy.random.RandomState()

        self.set_fluorophore(**config.fluorophore)  # => self.wave_length

        self.hc_const = config.hc_const

        self.set_shutter(**config.shutter)
        self.set_light_source(**config.light_source)
        self.set_dichroic_mirror(**config.dichroic_mirror)

        self.image_magnification = config.magnification
        _log.info('--- Magnification: x {}'.format(self.image_magnification))

        self.set_detector(**config.detector)
        self.set_analog_to_digital_converter(rng=rng, **config.analog_to_digital_converter)
        self.set_excitation_filter(**config.excitation_filter)
        self.set_emission_filter(**config.emission_filter)

        self.fluorophore_psf = self.get_PSF_detector()

    def set_shutter(self, start_time=None, end_time=None, time_open=None, time_lapse=None, switch=True):
        self.shutter_switch = switch
        self.shutter_start_time = start_time
        self.shutter_end_time = end_time
        self.shutter_time_open = time_open or end_time - start_time
        self.shutter_time_lapse = time_lapse or end_time - start_time

        _log.info('--- Shutter:')
        _log.info('    Start-Time = {} sec'.format(self.shutter_start_time))
        _log.info('    End-Time   = {} sec'.format(self.shutter_end_time))
        _log.info('    Time-open  = {} sec'.format(self.shutter_time_open))
        _log.info('    Time-lapse = {} sec'.format(self.shutter_time_lapse))

    def set_light_source(self, type=None, wave_length=None, flux_density=None, radius=None, angle=None, switch=True):
        self.source_switch = switch
        self.source_type = type
        self.source_wavelength = wave_length
        self.source_flux_density = flux_density
        self.source_radius = radius
        self.source_angle = angle

        _log.info('--- Light Source:{}'.format(self.source_type))
        _log.info('    Wave Length = {} m'.format(self.source_wavelength))
        _log.info('    Beam Flux Density = {} W/cm2'.format(self.source_flux_density))
        _log.info('    1/e2 Radius = {} m'.format(self.source_radius))
        _log.info('    Angle = {} degree'.format(self.source_angle))

    def set_fluorophore(
            self, type=None, wave_length=None, normalization=None, radius=None, width=None,
            min_wave_length=None, max_wave_length=None, radial_cutoff=None, depth_cutoff=None):
        self.radial = numpy.arange(0.0, radial_cutoff, 1e-9, dtype=float)
        self.depth = numpy.arange(0.0, depth_cutoff, 1e-9, dtype=float)
        self.wave_length = numpy.arange(min_wave_length, max_wave_length, 1e-9, dtype=float)
        self.wave_number = 2 * numpy.pi / (self.wave_length / 1e-9)  # 1/nm

        self.fluorophore_type = type
        self.fluorophore_radius = radius
        self.psf_normalization = normalization

        if type == 'Gaussian':
            N = len(self.wave_length)
            fluorophore_excitation = numpy.zeros(N, dtype=float)
            fluorophore_emission = numpy.zeros(N, dtype=float)
            idx = (numpy.abs(self.wave_length / 1e-9 - wave_length / 1e-9)).argmin()
            # idx = (numpy.abs(self.wave_length / 1e-9 - self.psf_wavelength / 1e-9)).argmin()
            fluorophore_excitation[idx] = 100
            fluorophore_emission[idx] = 100
            self.fluoex_eff = fluorophore_excitation
            self.fluoem_eff = fluorophore_emission

            fluorophore_excitation /= sum(fluorophore_excitation)
            fluorophore_emission /= sum(fluorophore_emission)
            self.fluoex_norm = fluorophore_excitation
            self.fluoem_norm = fluorophore_emission

            self.psf_wavelength = wave_length
            self.psf_width = width

        else:
            (fluorophore_excitation, fluorophore_emission) = io.read_fluorophore_catalog(type)
            fluorophore_excitation = self.calculate_efficiency(fluorophore_excitation, self.wave_length)
            fluorophore_emission = self.calculate_efficiency(fluorophore_emission, self.wave_length)
            fluorophore_excitation = numpy.array(fluorophore_excitation)
            fluorophore_emission = numpy.array(fluorophore_emission)
            index_ex = fluorophore_excitation.argmax()
            index_em = fluorophore_emission.argmax()
            fluorophore_excitation[index_ex] = 100
            fluorophore_emission[index_em] = 100
            self.fluoex_eff = fluorophore_excitation
            self.fluoem_eff = fluorophore_emission

            fluorophore_excitation /= sum(fluorophore_excitation)
            fluorophore_emission /= sum(fluorophore_emission)
            self.fluoex_norm = fluorophore_excitation
            self.fluoem_norm = fluorophore_emission

            if wave_length is not None:
                warnings.warn('The given wave length [{}] was ignored'.format(wave_length))
            self.psf_wavelength = self.wave_length[index_em]
            self.psf_width = None

        _log.info('--- Fluorophore: {} PSF'.format(self.fluorophore_type))
        _log.info('    Wave Length   =  {} m'.format(self.psf_wavelength))
        _log.info('    Normalization =  {}'.format(self.psf_normalization))
        _log.info('    Fluorophore radius =  {} m'.format(self.fluorophore_radius))
        if self.psf_width is not None:
            _log.info('    Lateral Width =  {} m'.format(self.psf_width[0]))
            _log.info('    Axial Width =  {} m'.format(self.psf_width[1]))
        _log.info('    PSF Normalization Factor =  {}'.format(self.psf_normalization))
        _log.info('    Emission  : Wave Length =  {} m'.format(self.psf_wavelength))

    def set_dichroic_mirror(self, type=None, switch=True):
        self.dichroic_switch = switch
        if type is not None:
            dichroic_mirror = io.read_dichroic_catalog(type)
            self.dichroic_eff = self.calculate_efficiency(dichroic_mirror, self.wave_length)
        else:
            self.dichroic_eff = numpy.zeros(len(self.wave_length), dtype=float)
        _log.info('--- Dichroic Mirror:')

    def set_detector(
            self, type=None, image_size=None, pixel_length=None, exposure_time=None, focal_point=None,
            QE=None, readout_noise=None, dark_count=None, emgain=None, switch=True, focal_norm=None):
        self.detector_switch = switch
        self.detector_type = type
        self.detector_image_size = image_size
        self.detector_pixel_length = pixel_length
        self.detector_exposure_time = exposure_time
        self.detector_focal_point = focal_point
        self.detector_qeff = QE
        self.detector_readout_noise = readout_noise
        self.detector_dark_count = dark_count
        self.detector_emgain = emgain
        # self.detector_base_position = base_position

        self.detector_focal_point = focal_point
        self.detector_focal_norm = focal_norm

        _log.info('--- Detector:  {}'.format(self.detector_type))
        if self.detector_image_size is not None:
            _log.info('    Image Size  =  {} x {}'.format(self.detector_image_size[0], self.detector_image_size[1]))
        _log.info('    Pixel Size  =  {} m/pixel'.format(self.detector_pixel_length))
        _log.info('    Focal Point =  {}'.format(self.detector_focal_point))
        _log.info('    Exposure Time =  {} sec'.format(self.detector_exposure_time))
        if self.detector_qeff is not None:
            _log.info('    Quantum Efficiency =  {} %'.format(100 * self.detector_qeff))
        _log.info('    Readout Noise =  {} electron'.format(self.detector_readout_noise))
        _log.info('    Dark Count =  {} electron/sec'.format(self.detector_dark_count))
        _log.info('    EM gain = x {}'.format(self.detector_emgain))
        # _log.info('    Position    =  {}'.format(self.detector_base_position))
        _log.info('Focal Center: {}'.format(self.detector_focal_point))
        _log.info('Normal Vector: {}'.format(self.detector_focal_norm))

    def set_analog_to_digital_converter(self, *, rng, bit=None, offset=None, fullwell=None, type=None, count=None):
        self.ADConverter_bit = bit
        self.ADConverter_fullwell = fullwell
        self.ADConverter_fpn_type = type
        self.ADConverter_fpn_count = count

        offset, gain = self.calculate_analog_to_digital_converter_gain(offset, rng)
        self.ADConverter_offset = offset
        self.ADConverter_gain = gain

        _log.info('--- A/D Converter: %d-bit' % (self.ADConverter_bit))
        # _log.info('    Gain = %.3f electron/count' % (self.ADConverter_gain))
        # _log.info('    Offset =  {} count'.format(self.ADConverter_offset))
        _log.info('    Fullwell =  {} electron'.format(self.ADConverter_fullwell))
        _log.info('    {}-Fixed Pattern Noise: {} count'.format(self.ADConverter_fpn_type, self.ADConverter_fpn_count))

    def calculate_analog_to_digital_converter_gain(self, ADC0, rng):
        # image pixel-size
        Nw_pixel = self.detector_image_size[0]
        Nh_pixel = self.detector_image_size[1]

        # set ADC parameters
        bit  = self.ADConverter_bit
        fullwell = self.ADConverter_fullwell
        # ADC0 = self.ADConverter_offset

        # set Fixed-Pattern noise
        FPN_type = self.ADConverter_fpn_type
        FPN_count = self.ADConverter_fpn_count

        # if FPN_type is None:
        if FPN_type == 'none':
            offset = numpy.full(Nw_pixel * Nh_pixel, ADC0)
        elif FPN_type == 'pixel':
            if rng is None:
                raise RuntimeError('A random number generator is required.')
            offset = numpy.rint(rng.normal(ADC0, FPN_count, Nw_pixel * Nh_pixel))
        elif FPN_type == 'column':
            column = rng.normal(ADC0, FPN_count, Nh_pixel)
            temporal = numpy.tile(column, (1, Nw_pixel))
            offset = numpy.rint(temporal.reshape(Nh_pixel * Nw_pixel))
        else:
            raise ValueError("FPN type [{}] is invalid ['pixel', 'column' or None]".format(FPN_type))

        # set ADC gain
        # gain = numpy.array(map(lambda x: (fullwell - 0.0) / (pow(2.0, bit) - x), offset))
        gain = (fullwell - 0.0) / (pow(2.0, bit) - offset)

        offset = offset.reshape([Nw_pixel, Nh_pixel])
        gain = gain.reshape([Nw_pixel, Nh_pixel])

        return (offset, gain)

    def set_excitation_filter(self, type=None, switch=True):
        self.excitation_switch = switch
        if type is not None:
            excitation_filter = io.read_excitation_catalog(type)
            self.excitation_eff = self.calculate_efficiency(excitation_filter, self.wave_length)
        else:
            self.excitation_eff = numpy.zeros(len(self.wave_length), dtype=float)
        _log.info('--- Excitation Filter:')

    def set_emission_filter(self, type=None, switch=True):
        self.emission_switch = switch
        if type is not None:
            emission_filter = io.read_emission_catalog(type)
            self.emission_eff = self.calculate_efficiency(emission_filter, self.wave_length)
        else:
            self.emission_eff = numpy.zeros(len(self.wave_length), dtype=float)
        _log.info('--- Emission Filter:')

    def get_PSF_detector(self):
        # Fluorophores Emission Intensity (wave_length)
        I = self.fluoem_norm

        # Photon Transmission Efficiency
        if self.dichroic_switch:
            I = I * 0.01 * self.dichroic_eff

        if self.emission_switch:
            I = I * 0.01 * self.emission_eff

        # PSF: Fluorophore
        if self.fluorophore_type == 'Gaussian':
            # For normalization
            # norm = list(map(lambda x: True if x > 1e-4 else False, I))
            norm = (I > 1e-4)

            # Ir = sum(list(map(lambda x: x * numpy.exp(-0.5 * (r / self.psf_width[0]) ** 2), norm)))
            # Iz = sum(list(map(lambda x: x * numpy.exp(-0.5 * (z / self.psf_width[1]) ** 2), norm)))
            Ir = norm * numpy.exp(-0.5 * numpy.power(self.radial / self.psf_width[0], 2))
            Iz = norm * numpy.exp(-0.5 * numpy.power(self.radial / self.psf_width[1], 2))

            psf_fl = numpy.sum(I) * numpy.array(list(map(lambda x: Ir * x, Iz)))

        else:
            # make the norm and wave_length array shorter
            psf_fl = numpy.sum(I) * self.get_PSF_fluorophore(self.radial, self.depth, self.psf_wavelength)

        psf_fl *= self.psf_normalization
        return psf_fl

    @staticmethod
    def get_PSF_fluorophore(r, z, wave_length):
        # set Magnification of optical system
        # M = self.image_magnification

        # set Numerical Appature
        NA = 1.4  # self.objective_NA

        # set alpha and gamma consts
        k = 2.0 * numpy.pi / (wave_length / 1e-9)
        alpha = k * NA
        gamma = k * numpy.power(NA / 2, 2)

        # set rho parameters
        N = 100
        drho = 1.0 / N
        # rho = numpy.array([(i + 1) * drho for i in range(N)])
        rho = numpy.arange(1, N + 1) * drho

        J0 = numpy.array(list(map(lambda x: j0(x * alpha * rho), r / 1e-9)))
        Y  = numpy.array(list(map(lambda x: numpy.exp(-2 * 1.j * x * gamma * rho * rho) * rho * drho, z / 1e-9)))
        I  = numpy.array(list(map(lambda x: x * J0, Y)))
        I_sum = I.sum(axis=2)

        # set PSF
        psf = numpy.array(list(map(lambda x: abs(x) ** 2, I_sum)))
        return psf

    @staticmethod
    def calculate_efficiency(data, wave_length):
        data = numpy.array(data, dtype = 'float')
        data = data[data[: , 0] % 1 == 0, :]

        efficiency = numpy.zeros(len(wave_length))

        wave_length = numpy.round(wave_length / 1e-9).astype(int)  #XXX: numpy.array(dtype=int)
        idx1 = numpy.in1d(wave_length, data[:, 0])
        idx2 = numpy.in1d(numpy.array(data[:, 0]), wave_length)

        efficiency[idx1] = data[idx2, 1]

        return efficiency.tolist()

def dict_merge(dct, merge_dct):
    for k, v in merge_dct.items():
        if isinstance(dct.get(k), dict) and isinstance(v, collections.Mapping):
            dict_merge(dct[k], v)
        else:
            dct[k] = v

class Configuration(collections.abc.Mapping):

    def __init__(self, filename=None, yaml=None):
        if filename is not None:
            assert yaml is None
            import yaml
            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader
            with open(filename) as f:
                self.__yaml = yaml.load(f.read(), Loader=Loader)
        elif yaml is not None:
            self.__yaml = yaml
        else:
            self.__yaml = None

    def __repr__(self):
        import yaml
        try:
            from yaml import CDumper as Dumper
        except ImportError:
            from yaml import Dumper
        return yaml.dump(self.__yaml, default_flow_style=False, Dumper=Dumper)

    def update(self, conf):
        if isinstance(conf, Configuration):
            dict_merge(self.__yaml, conf.yaml)
        elif isinstance(conf, dict):
            dict_merge(self.__yaml, conf)
        else:
            import yaml
            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader
            dict_merge(self.__yaml, yaml.load(conf, Loader=Loader))

    @property
    def yaml(self):
        return self.__yaml

    def get(self, key, defaultobj=None):
        return self.__yaml.get(key, defaultobj)

    def __getitem__(self, key):
        value = self.__yaml[key]
        if isinstance(value, dict):
            assert 'value' in value
            return value['value']
        return value

    def __len__(self):
        return len(self.__yaml)

    def __iter__(self):
        return (key for key, value in self.__yaml.items() if not isinstance(value, dict) or 'value' in value)
        # return iter(self.__yaml)

    def __getattr__(self, key):
        assert key in self.__yaml
        value = self.__yaml[key]
        if isinstance(value, dict):
            if 'value' not in value:
                return Configuration(yaml=value)
            else:
                return value['value']
        return value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return

        assert key in self.__yaml
        assert not isinstance(value, dict)
        value_ = self.__yaml[key]
        if isinstance(value_, dict):
            if 'value' not in value_:
                return Configuration(yaml=value_)
            else:
                value_['value'] = value
        self.__yaml[key] = value

class Image(object):

    def __init__(self, data):
        assert data.ndim == 2
        self.__data = data

    def as_array(self):
        return self.__data

    def save(self, filename, low=None, high=None):
        data = self.as_array()
        low = data.min() if low is None else low
        high = data.max() if high is None else high
        data = (high - data) / (high - low) * 255
        data = numpy.uint8(self.__data)

        import PIL.Image
        img = PIL.Image.fromarray(data)
        img.save(filename)

    def plot(self):
        import matplotlib.pylab as plt
        fig, ax = plt.subplots()
        plt.imshow(self.__data, interpolation='none', cmap="gray")
        plt.show()

    def _ipython_display_(self):
        """
        Displays the object as a side effect.
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        self.plot()

class EPIFMSimulator(object):

    def __init__(self, config=None):
        """ Constructor

        Args:
            config (Configuration or str, optional): Configurations.
                The default is None.
        """
        if config is None:
            config = Configuration()
        elif isinstance(config, str):
            config = Configuration(filename=config)
        elif not isinstance(config, Configuration):
            raise TypeError("Configuration or str must be given [{}].".format(type(config)))

        self.__config = config
        # # Config
        # self.__config = scopyon.config.Config()
        # self.__config.update('hc_const', config["epifm"]["hc_const"]
        # self.__config.update('shutter_switch', config["epifm"]["shutter"]["switch"])
        # self.__config.update('shutter_start_time', config["epifm"]["shutter"]["start_time"])
        # self.__config.update('shutter_end_time', config["epifm"]["shutter"]["end_time"])
        # self.__config.update('shutter_time_open', config["epifm"]["shutter"]["time_open"])
        # self.__config.update('shutter_time_lapse', config["epifm"]["shutter"]["time_lapse"])
        # self.__config.update('source_switch', config["epifm"]["light_source"]["switch"])
        # self.__config.update('source_type', config["epifm"]["light_source"]["type"])
        # self.__config.update('source_wavelength', config["epifm"]["light_source"]["wave_length"])
        # self.__config.update('source_flux_density', config["epifm"]["light_source"]["flux_density"])
        # self.__config.update('source_radius', config["epifm"]["light_source"]["radius"])
        # self.__config.update('source_angle', config["epifm"]["light_source"]["angle"])
        # self.__config.update('fluorophore_type', config["epifm"]["fluorophore"]["type"])
        # self.__config.update('fluorophore_radius', config["epifm"]["fluorophore"]["radius"])
        # self.__config.update('psf_wavelength', config["epifm"]["fluorophore"]["wave_length"])
        # self.__config.update('psf_min_wave_length', config["epifm"]["fluorophore"]["min_wave_length"])
        # self.__config.update('psf_max_wave_length', config["epifm"]["fluorophore"]["min_wave_length"])
        # self.__config.update('psf_normalization', config["epifm"]["fluorophore"]["normalization"])
        # self.__config.update('psf_width', config["epifm"]["fluorophore"]["width"])
        # # self.__config.update('psf_cutoff', config["epifm"]["fluorophore"]["cutoff"])  # deprecated
        # self.__config.update('psf_radial_cutoff', config["epifm"]["fluorophore"]["radial_cutoff"])
        # self.__config.update('psf_depth_cutoff', config["epifm"]["fluorophore"]["depth_cutoff"])
        # self.__config.update('psf_file_name_format', config["epifm"]["fluorophore"]["file_name_format"])
        # self.__config.update('dichroic_switch', config["epifm"]["dichroic_mirror"]["switch"])
        # self.__config.update('dichroic_mirror', config["epifm"]["dichroic_mirror"]["type"])
        # self.__config.update('image_magnification', config["epifm"]["magnification"])
        # self.__config.update('detector_switch', config["epifm"]["detector"]["switch"])
        # self.__config.update('detector_type', config["epifm"]["detector"]["type"])
        # self.__config.update('detector_image_size', config["epifm"]["detector"]["image_size"])
        # self.__config.update('detector_pixel_length', config["epifm"]["detector"]["pixel_length"])
        # self.__config.update('detector_exposure_time', config["epifm"]["detector"]["exposure_time"])
        # self.__config.update('detector_focal_point', numpy.array(config["epifm"]["detector"]["focal_point"]))  #XXX
        # self.__config.update('detector_qeff', config["epifm"]["detector"]["QE"])
        # self.__config.update('detector_readout_noise', config["epifm"]["detector"]["readout_noise"])
        # self.__config.update('detector_dark_count', config["epifm"]["detector"]["dark_count"])
        # self.__config.update('detector_emgain', config["epifm"]["detector"]["emgain"])
        # self.__config.update('ADConverter_bit', config["epifm"]["analog_to_digital_converter"]["bit"])
        # self.__config.update('ADConverter_offset', config["epifm"]["analog_to_digital_converter"]["offset"])
        # self.__config.update('ADConverter_fullwell', config["epifm"]["analog_to_digital_converter"]["fullwell"])
        # self.__config.update('ADConverter_fpn_type', config["epifm"]["analog_to_digital_converter"]["type"])
        # self.__config.update('ADConverter_fpn_count', config["epifm"]["analog_to_digital_converter"]["count"])
        # # self.__config.update('detector_focal_point', config["epifm"]["illumination_path"]["focal_point"])
        # self.__config.update('detector_focal_norm', config["epifm"]["illumination_path"]["focal_norm"])
        # self.__config.update('excitation_switch', config["epifm"]["excitation_filter"]["switch"])
        # self.__config.update('excitation_filter', config["epifm"]["excitation_filter"]["type"])
        # self.__config.update('emission_switch', config["epifm"]["emission_filter"]["switch"])
        # self.__config.update('emission_filter', config["epifm"]["emission_filter"]["type"])

        # self.__config.update('background_switch', config["effects"]["background"]["switch"])
        # self.__config.update('background_mean', config["effects"]["background"]["mean"])
        # self.__config.update('crosstalk_switch', config["effects"]["crosstalk"]["switch"])
        # self.__config.update('crosstalk_width', config["effects"]["crosstalk"]["width"])
        # self.__config.update('quantum_yield', config["effects"]["fluorescence"]["quantum_yield"])
        # self.__config.update('abs_coefficient', config["effects"]["fluorescence"]["abs_coefficient"])
        # self.__config.update('photobleaching_switch', config["effects"]["photo_bleaching"]["switch"])
        # self.__config.update('photobleaching_tau0', config["effects"]["photo_bleaching"]["tau0"])
        # self.__config.update('photobleaching_alpha', config["effects"]["photo_bleaching"]["alpha"])
        # self.__config.update('photoactivation_switch', config["effects"]["photo_activation"]["switch"])
        # self.__config.update('photoactivation_turn_on_ratio', config["effects"]["photo_activation"]["turn_on_ratio"])
        # self.__config.update('photoactivation_activation_yield', config["effects"]["photo_activation"]["activation_yield"])
        # self.__config.update('photoactivation_frac_preactivation', config["effects"]["photo_activation"]["frac_preactivation"])
        # self.__config.update('photoblinking_switch', config["effects"]["photo_blinking"]["switch"])
        # self.__config.update('photoblinking_t0_on', config["effects"]["photo_blinking"]["t0_on"])
        # self.__config.update('photoblinking_t0_off', config["effects"]["photo_blinking"]["t0_off"])
        # self.__config.update('photoblinking_a_on', config["effects"]["photo_blinking"]["a_on"])
        # self.__config.update('photoblinking_a_off', config["effects"]["photo_blinking"]["a_off"])

    def form_image(self, inputs, rng=None, debug=False):
        """Form image.

        Args:
            inputs (array_like): A list of points. The shape must be '(n, 3)', where
                'n' means the number of points.
            rng (numpy.RandomState, optional): A random number generator.
                The default is None.

        Returns:
            Image: An image.
        """
        if isinstance(inputs, numpy.ndarray):
            if inputs.ndim != 2:
                raise ValueError("The given 'inputs' have wrong dimension.")

            if inputs.shape[1] == 2:
                data = numpy.hstack((
                    numpy.zeros((inputs.shape[0], 1)),
                    inputs))
            elif inputs.shape[1] == 3:
                origin = numpy.array(self.__config.preprocessing.origin)
                data = inputs - origin
                unit_z = numpy.cross(
                    self.__config.preprocessing.unit_x,
                    self.__config.preprocessing.unit_y)
                data = numpy.hstack((
                    numpy.dot(data, unit_z).reshape((-1, 1)),
                    numpy.dot(data, self.__config.preprocessing.unit_x).reshape((-1, 1)),
                    numpy.dot(data, self.__config.preprocessing.unit_y).reshape((-1, 1))))
            else:
                raise ValueError("The given 'inputs' have wrong shape.")

            data = numpy.hstack((
                data,
                numpy.zeros((data.shape[0], 5))))
            data[:, 6] = 1.0
            data = ((0.0, data), )
        else:
            raise TypeError(
                    "Invalid argument was given [{}]."
                    " A ndarray is expected.".format(type(inputs)))

        # base = scopyon.epifm.EPIFMSimulator(self.__config, rng=rng)
        base = scopyon.epifm.EPIFMSimulator(
                configs=_EPIFMConfigs(self.__config.default.epifm, rng=rng),
                effects=PhysicalEffects(self.__config.default.effects))

        camera, true_data = base.output_frame(data, rng=rng)
        # camera[:, :, 0] => expected
        # camera[:, :, 1] => ADC
        img = Image(camera[:, :, 1])
        if debug:
            return img, (camera, true_data)
        return img

def create_simulator(config=None, method=None):
    """Return a simulator.

    Args:
        config (Configuration, optional): Configurations.
        method (str, optional): A name of method used.
            The default is None ('epifm').

    Returns:
        scopyon.epifm.EPIFMSimulator: A simulator
    """
    DEFAULT_METHOD = 'epifm'
    if method is None:
        if config is not None:
            method = config.get('method', DEFAULT_METHOD).lower()
        else:
            method = DEFAULT_METHOD
    if method == 'epifm':
        return EPIFMSimulator(config=config)
    else:
        raise ValueError(f"An unknown method [{method}] was given.")

def form_image(inputs, *, method=None, config=None, rng=None, debug=False):
    """Form image.

    Args:
        inputs (array_like): A list of points. The shape must be '(n, 3)',
            where 'n' means the number of points.
        method (str, optional): A name of method used.
            The default is None ('epifm').
        config (Configuration, optional): Configurations.
            The default is None.
        rng (numpy.RandomState, optional): A random number generator.
            The default is None.

    Returns:
        Image: An image
    """
    sim = create_simulator(config, method=method)
    return sim.form_image(inputs, rng=rng, debug=debug)


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    config = Configuration(filename)
    print(config.yaml)
    n = 10
    inputs = numpy.hstack((numpy.random.uniform(0, 5e-5, size=(n, 2)), numpy.zeros((n, 1)), ))
    print(form_image(inputs, config=config))
