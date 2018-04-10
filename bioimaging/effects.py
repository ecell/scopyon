import sys
import os
import copy
import tempfile
import math
import random
#import h5py
import ctypes
import multiprocessing
import functools

#import pylab
import scipy
import numpy

from . import parameter_effects

from scipy.special import j0
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.misc    import toimage

from logging import getLogger
_log = getLogger(__name__)

IMAGE_SIZE_LIMIT=3000


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

    def __init__(self, user_configs_dict = None):
        # default setting
        configs_dict = parameter_effects.__dict__.copy()

        # user setting
        if user_configs_dict is not None:
            if type(user_configs_dict) != type({}):
                _log.info('Illegal argument type for constructor of Configs class')
                sys.exit()
            configs_dict.update(user_configs_dict)

        for key, val in configs_dict.items():
            if key[0] != '_': # Data skip for private variables in setting_dict.
                if type(val) == type({}) or type(val) == type([]):
                    copy_val = copy.deepcopy(val)
                else:
                    copy_val = val
                setattr(self, key, copy_val)

    def _set_data(self, key, val):
        if val != None:
            setattr(self, key, val)

    def set_background(self, mean=None):
        _log.info('--- Background: ')

        self._set_data('background_switch', True)
        self._set_data('background_mean', mean)

        _log.info('    Mean = {} photons'.format(self.background_mean))

    def set_crosstalk(self, width=None):
        _log.info('--- Photoelectron Crosstalk: ')

        self._set_data('crosstalk_switch', True)
        self._set_data('crosstalk_width', width)

        _log.info('    Width = {} pixels'.format(self.crosstalk_width))

    def set_fluorescence(self, quantum_yield=None,
                                abs_coefficient=None):
        _log.info('--- Fluorescence: ')

        self._set_data('quantum_yield', quantum_yield)
        self._set_data('abs_coefficient', abs_coefficient)

        _log.info('    Quantum Yield =  {}'.format(self.quantum_yield))
        _log.info('    Abs. Coefficient =  {} 1/(M cm)'.format(self.abs_coefficient))
        _log.info('    Abs. Cross-section =  {} cm^2'.format((numpy.log(10)*self.abs_coefficient*0.1/6.022e+23)*1e+4))

    def set_photobleaching(self, tau0=None,
                                alpha=None):
        _log.info('--- Photobleaching: ')

        self._set_data('photobleaching_switch', True)
        self._set_data('photobleaching_tau0', tau0)
        self._set_data('photobleaching_alpha', alpha)

        _log.info('    Photobleaching tau0  =  {}'.format(self.photobleaching_tau0))
        _log.info('    Photobleaching alpha =  {}'.format(self.photobleaching_alpha))

    def set_photoactivation(self, turn_on_ratio=None,
                                activation_yield=None,
                                frac_preactivation=None):
        _log.info('--- Photoactivation: ')

        self._set_data('photoactivation_switch', True)
        self._set_data('photoactivation_turn_on_ratio', turn_on_ratio)
        self._set_data('photoactivation_activation_yield', activation_yield)
        self._set_data('photoactivation_frac_preactivation', frac_preactivation)

        _log.info('    Turn-on Ratio  =  {}'.format(self.photoactivation_turn_on_ratio))
        _log.info('    Effective Ratio  =  {}'.format(activation_yield*turn_on_ratio/(1 + frac_preactivation*turn_on_ratio)))
        _log.info('    Reaction Yield =  {}'.format(self.photoactivation_activation_yield))
        _log.info('    Fraction of Preactivation =  {}'.format(self.photoactivation_frac_preactivation))

    def set_photoblinking(self, t0_on=None, a_on=None,
                                t0_off=None, a_off=None):
        _log.info('--- Photo-blinking: ')

        self._set_data('photoblinking_switch', True)
        self._set_data('photoblinking_t0_on', t0_on)
        self._set_data('photoblinking_a_on', a_on)
        self._set_data('photoblinking_t0_off', t0_off)
        self._set_data('photoblinking_a_off', a_off)

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

    def get_photobleaching_property(self, dt, n_emit0):
        # set the photobleaching-time
        tau0  = self.photobleaching_tau0
        alpha = self.photobleaching_alpha

        # set photon budget
        photon0 = (tau0/dt)*n_emit0

        tau_bleach = numpy.array([j*dt + tau0 for j in range(int(1e+7))])
        p_bleach = self.get_prob_bleach(tau_bleach, dt)

        # get photon budget and photobleaching-time
        tau = numpy.random.choice(tau_bleach, 1, p=p_bleach)
        budget = (tau/tau0)*photon0

        return tau, budget

    def get_photophysics_for_epifm(self, delta, n_emit0, N_part, interval, rng=None):
        state = numpy.zeros(shape=(len(delta)))
        dt = delta[0]

        if (self.photobleaching_switch == True):
            # set the photobleaching-time
            tau0  = self.photobleaching_tau0
            alpha = self.photobleaching_alpha
            prob_func = functools.partial(levy_probability_function, t0=tau0, a=alpha)

            # set photon budget
            photon0 = (tau0 / interval) * n_emit0

            tau_bleach = numpy.array([j * interval + tau0 for j in range(int(1e+7))])
            prob = prob_func(tau_bleach)
            norm = prob.sum()
            p_bleach = prob / norm

            # get photon budget and photobleaching-time
            if rng is None:
                raise RuntimeError('A random number generator is required.')
            tau = rng.choice(tau_bleach, N_part, p=p_bleach)
        else:
            raise RuntimeError('Not supported')

        # sequences
        budget = []
        state  = []

        for i in range(N_part):
            # get photon budget and photobleaching-time
            photons = (tau[i] / tau0) * photon0

            # bleaching time
            Ni = (numpy.abs(numpy.cumsum(delta) - tau[i])).argmin()

            state_bleach = numpy.zeros(shape=(len(delta)))
            state_bleach[0: Ni] = numpy.ones(shape=(Ni))

            # set photon budget and fluorescence state
            budget.append(photons)
            state.append(state_bleach)

        return numpy.array(state), numpy.array(budget)

    def set_photophysics_4palm(self, start, end, dt, f, F, N_part):
        ##### PALM Configuration
        NNN = int(1e+7)

        # set the photobleaching-time
        tau0  = self.photobleaching_tau0
        alpha = self.photobleaching_alpha

        # set photon budget
        #photon0 = (tau0/dt)*N_emit0
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
        numpy.random.seed()

        # get photon budget and photobleaching-time
        tau = numpy.random.choice(tau_bleach, N_part, p=p_bleach)

        for i in range(N_part):

            # get photon budget and photobleaching-time
            photons = (tau[i]/tau0)*photon0

            N = int((end - start)/dt)

            time  = numpy.array([j*dt + start for j in range(N)])
            state = numpy.zeros(shape=(N))

            #### Photoactivation: overall activation yeild
            p = self.photoactivation_activation_yield
            q = self.photoactivation_frac_preactivation

            r = numpy.random.uniform(0, 1, N/F*f)
            s = (r < p).astype('int')

            index = numpy.abs(s - 1).argmin()
            t0 = (index/f*F + numpy.remainder(index, f))*dt + start

            t1 = t0 + tau[i]

            N0 = (numpy.abs(time - t0)).argmin()
            N1 = (numpy.abs(time - t1)).argmin()
            state[N0:N1] = numpy.ones(shape=(N1-N0))

            if (self.photoblinking_switch == True):

                # ON-state
                t_on  = numpy.random.choice(tau_on,  100, p=p_on)
                # OFF-state
                t_off = numpy.random.choice(tau_off, 100, p=p_off)

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

    def set_photophysics_4lscm(self, start, end, dt, N_part):
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
        numpy.random.seed()

        # get photon budget and photobleaching-time
        tau = numpy.random.choice(tau_bleach, N_part, p=p_bleach)

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
                t_on  = numpy.random.choice(tau_on,  100, p=p_on)
                # OFF-state
                t_off = numpy.random.choice(tau_off, 100, p=p_off)

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
