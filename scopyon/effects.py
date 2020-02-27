import functools

import numpy

from . import constants

from logging import getLogger
_log = getLogger(__name__)


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

    def __init__(self):
        pass

    def initialize(self, config):
        self.fluorescence_bleach = []
        self.fluorescence_budget = []
        self.fluorescence_state  = []

        self.set_background(mean=config.background_mean, switch=config.background_switch)

        _log.info('--- Background: ')
        _log.info('    Mean = {} photons'.format(self.background_mean))

        self.set_fluorescence(quantum_yield=config.quantum_yield, abs_coefficient=config.abs_coefficient)

        _log.info('--- Fluorescence: ')
        _log.info('    Quantum Yield =  {}'.format(self.quantum_yield))
        _log.info('    Abs. Coefficient =  {} 1/(M cm)'.format(self.abs_coefficient))
        _log.info('    Abs. Cross-section =  {} cm^2'.format((numpy.log(10) * self.abs_coefficient * 0.1 / constants.N_A) * 1e+4))

        self.set_photobleaching(tau0=config.photobleaching_tau0, alpha=config.photobleaching_alpha, switch=config.photobleaching_switch)

        _log.info('--- Photobleaching: ')
        _log.info('    Photobleaching tau0  =  {}'.format(self.photobleaching_tau0))
        _log.info('    Photobleaching alpha =  {}'.format(self.photobleaching_alpha))

        self.set_photoactivation(turn_on_ratio=config.photoactivation_turn_on_ratio, activation_yield=config.photoactivation_activation_yield, frac_preactivation=config.photoactivation_frac_preactivation, switch=config.photoactivation_switch)

        _log.info('--- Photoactivation: ')
        _log.info('    Turn-on Ratio  =  {}'.format(self.photoactivation_turn_on_ratio))
        _log.info('    Effective Ratio  =  {}'.format(self.photoactivation_activation_yield * self.photoactivation_turn_on_ratio / (1 + self.photoactivation_frac_preactivation * self.photoactivation_turn_on_ratio)))
        _log.info('    Reaction Yield =  {}'.format(self.photoactivation_activation_yield))
        _log.info('    Fraction of Preactivation =  {}'.format(self.photoactivation_frac_preactivation))

        self.set_photoblinking(t0_on=config.photoblinking_t0_on, a_on=config.photoblinking_a_on, t0_off=config.photoblinking_t0_off, a_off=config.photoblinking_a_off, switch=config.photoblinking_switch)

        _log.info('--- Photo-blinking: ')
        _log.info('    (ON)  t0 =  {} sec'.format(self.photoblinking_t0_on))
        _log.info('    (ON)  a  =  {}'.format(self.photoblinking_a_on))
        _log.info('    (OFF) t0 =  {} sec'.format(self.photoblinking_t0_off))
        _log.info('    (OFF) a  =  {}'.format(self.photoblinking_a_off))

    def _set_data(self, key, val):
        if val != None:
            setattr(self, key, val)

    def set_background(self, mean=None, switch=True):
        self._set_data('background_switch', switch)
        self._set_data('background_mean', mean)

    def set_fluorescence(self, quantum_yield=None, abs_coefficient=None):
        self._set_data('quantum_yield', quantum_yield)
        self._set_data('abs_coefficient', abs_coefficient)

    def set_photobleaching(self, tau0=None, alpha=None, switch=True):
        self._set_data('photobleaching_switch', switch)
        self._set_data('photobleaching_tau0', tau0)
        self._set_data('photobleaching_alpha', alpha)

    def set_photoactivation(self, turn_on_ratio=None, activation_yield=None, frac_preactivation=None, switch=True):
        self._set_data('photoactivation_switch', switch)
        self._set_data('photoactivation_turn_on_ratio', turn_on_ratio)
        self._set_data('photoactivation_activation_yield', activation_yield)
        self._set_data('photoactivation_frac_preactivation', frac_preactivation)

    def set_photoblinking(self, t0_on=None, a_on=None, t0_off=None, a_off=None, switch=True):
        self._set_data('photoblinking_switch', switch)
        self._set_data('photoblinking_t0_on', t0_on)
        self._set_data('photoblinking_a_on', a_on)
        self._set_data('photoblinking_t0_off', t0_off)
        self._set_data('photoblinking_a_off', a_off)

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
