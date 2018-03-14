
import sys
import os
import copy
import tempfile
import math
import operator
import random
#import h5py
import ctypes
import multiprocessing

#import pylab
import scipy
import numpy

from . import parameter_effects

from time import sleep

from scipy.special import j0
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.misc    import toimage

IMAGE_SIZE_LIMIT=3000

class PhysicalEffects() :

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
                print('Illegal argument type for constructor of Configs class')
                sys.exit()
            configs_dict.update(user_configs_dict)

        for key, val in configs_dict.items():
            if key[0] != '_': # Data skip for private variables in setting_dict.
                if type(val) == type({}) or type(val) == type([]):
                    copy_val = copy.deepcopy(val)
                else:
                    copy_val = val
                setattr(self, key, copy_val)


    def _set_data(self, key, val) :

        if val != None:
            setattr(self, key, val)



    def set_background(self, mean=None) :

        print('--- Background :')

        self._set_data('background_switch', True)
        self._set_data('background_mean', mean)

        print('\tMean = ', self.background_mean, 'photons')



    def set_crosstalk(self, width=None) :

        print('--- Photoelectron Crosstalk :')

        self._set_data('crosstalk_switch', True)
        self._set_data('crosstalk_width', width)

        print('\tWidth = ', self.crosstalk_width, 'pixels')



    def set_fluorescence(self, quantum_yield=None,
                                abs_coefficient=None) :

        print('--- Fluorescence :')

        self._set_data('quantum_yield', quantum_yield)
        self._set_data('abs_coefficient', abs_coefficient)

        print('\tQuantum Yield = ', self.quantum_yield)
        print('\tAbs. Coefficient = ', self.abs_coefficient, '1/(M cm)')
        print('\tAbs. Cross-section = ', (numpy.log(10)*self.abs_coefficient*0.1/6.022e+23)*1e+4, 'cm^2')



    def set_photobleaching(self, tau0=None,
                                alpha=None) :

        print('--- Photobleaching :')

        self._set_data('photobleaching_switch', True)
        self._set_data('photobleaching_tau0', tau0)
        self._set_data('photobleaching_alpha', alpha)

        print('\tPhotobleaching tau0  = ', self.photobleaching_tau0)
        print('\tPhotobleaching alpha = ', self.photobleaching_alpha)



    def set_photoactivation(self, turn_on_ratio=None,
                                activation_yield=None,
                                frac_preactivation=None) :

        print('--- Photoactivation :')

        self._set_data('photoactivation_switch', True)
        self._set_data('photoactivation_turn_on_ratio', turn_on_ratio)
        self._set_data('photoactivation_activation_yield', activation_yield)
        self._set_data('photoactivation_frac_preactivation', frac_preactivation)

        print('\tTurn-on Ratio  = ', self.photoactivation_turn_on_ratio)
        print('\tEffective Ratio  = ', activation_yield*turn_on_ratio/(1 + frac_preactivation*turn_on_ratio))
        print('\tReaction Yield = ', self.photoactivation_activation_yield)
        print('\tFraction of Preactivation = ', self.photoactivation_frac_preactivation)



    def set_photoblinking(self, t0_on=None, a_on=None,
                                t0_off=None, a_off=None) :

        print('--- Photo-blinking : ')

        self._set_data('photoblinking_switch', True)
        self._set_data('photoblinking_t0_on', t0_on)
        self._set_data('photoblinking_a_on', a_on)
        self._set_data('photoblinking_t0_off', t0_off)
        self._set_data('photoblinking_a_off', a_off)

        print('\t(ON)  t0 = ', self.photoblinking_t0_on, 'sec')
        print('\t(ON)  a  = ', self.photoblinking_a_on)
        print('\t(OFF) t0 = ', self.photoblinking_t0_off, 'sec')
        print('\t(OFF) a  = ', self.photoblinking_a_off)



    def prob_levy(self, t, t0, a) :

        prob = (a/t0)*(t0/t)**(1+a)
        #prob = (t0/t)**(1+a)

        return prob



    def get_prob_bleach(self, tau, dt) :

        # set the photobleaching-time
        tau0  = self.photobleaching_tau0
        alpha = self.photobleaching_alpha

        # set probability
        prob = self.prob_levy(tau, tau0, alpha)
        norm = prob.sum()
        p_bleach = prob/norm

        return p_bleach



    def get_prob_blink(self, tau_on, tau_off, dt) :

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



    def get_photobleaching_property(self, dt, n_emit0) :

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



    def set_photophysics_4epifm(self, time, delta, n_emit0, N_part) :

        state = numpy.zeros(shape=(len(delta)))
        dt = delta[0]

        if (self.photobleaching_switch == True) :
            # set the photobleaching-time
            tau0  = self.photobleaching_tau0
            alpha = self.photobleaching_alpha

            # set photon budget
            photon0 = (tau0/dt)*n_emit0

            tau_bleach = numpy.array([j*dt+tau0 for j in range(int(1e+7))])
            prob = self.prob_levy(tau_bleach, tau0, alpha)
            norm = prob.sum()
            p_bleach = prob/norm

            # get random number
            numpy.random.seed()

            # get photon budget and photobleaching-time
            tau = numpy.random.choice(tau_bleach, N_part, p=p_bleach)


        # sequences
        budget = []
        state  = []

        for i in range(N_part) :

            # get photon budget and photobleaching-time
            photons = (tau[i]/tau0)*photon0

            # bleaching time
            state_bleach = numpy.zeros(shape=(len(time)))

            Ni = (numpy.abs(numpy.cumsum(delta) - tau[i])).argmin()

            state_bleach[0:Ni] = numpy.ones(shape=(Ni))

            # set photon budget and fluorescence state
            budget.append(photons)
            state.append(state_bleach)

        #####
        self.fluorescence_budget = numpy.array(budget)
        self.fluorescence_state  = numpy.array(state)



    def set_photophysics_4palm(self, start, end, dt, f, F, N_part) :

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
        if (self.photoblinking_switch == True) :

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

        for i in range(N_part) :

            # get photon budget and photobleaching-time
            photons = (tau[i]/tau0)*photon0

            N = int((end - start)/dt)

            time  = numpy.array([j*dt + start for j in range(N)])
            state = numpy.zeros(shape=(N))

            #### Photoactivation : overall activation yeild
            p = self.photoactivation_activation_yield
            q = self.photoactivation_frac_preactivation

            r = numpy.random.uniform(0, 1, N/F*f)
            s = (r < p).astype('int')

            index = numpy.abs(s - 1).argmin()
            t0 = (index/f*F + numpy.remainder(index, f))*dt + start

#           t0 = end
#
#           for k in range(N/F) :
#               for l in range(f) :
#                   # random
#                   r = numpy.random.uniform(0, 1)
#
#                   if (r < PE) :
#                       t0 = (k + l)*dt
#                       break
#
#               if (t0 < end) : break

            t1 = t0 + tau[i]

            N0 = (numpy.abs(time - t0)).argmin()
            N1 = (numpy.abs(time - t1)).argmin()
            state[N0:N1] = numpy.ones(shape=(N1-N0))

            if (self.photoblinking_switch == True) :

                # ON-state
                t_on  = numpy.random.choice(tau_on,  100, p=p_on)
                # OFF-state
                t_off = numpy.random.choice(tau_off, 100, p=p_off)

                k = (numpy.abs(numpy.cumsum(t_on) - tau[i])).argmin()

                # merge t_on/off arrays
                t_blink = numpy.array([t_on[0:k], t_off[0:k]])
                t = numpy.reshape(t_blink, 2*k, order='F')

                i_on  = N0

                for j in range(k) :

                    f_on  = i_on  + int(t[j]/dt)
                    i_off = f_on
                    f_off = i_off + int(t[j+1]/dt)

                    state[i_on:f_on]   = 1
                    state[i_off:f_off] = 0

                    i_on = f_off

#               time_blinking = numpy.cumsum(t)
#
#               # set state array
#               s_on  = numpy.ones(shape=(k))
#               s_off = numpy.zeros(shape=(k))
#               s = numpy.array([s_on, s_off])
#
#               state_blinking = numpy.reshape(s, 2*k, order='F')
#               state_blinking = numpy.reshape(s, 2*k, order='F')
#
#               # set fluorescence state (with photoblinking)
#               budget.append(photons)
#               time_act.append(time_blinking)
#               state_act.append(state_blinking)

            # set fluorescence state (without photoblinking)
            budget.append(photons)
            state_act.append(state)

        self.fluorescence_bleach = numpy.array(tau)
        self.fluorescence_budget = numpy.array(budget)
        self.fluorescence_state  = numpy.array(state_act)



    def set_photophysics_4lscm(self, start, end, dt, N_part) :

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
        if (self.photoblinking_switch == True) :

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

        for i in range(N_part) :

            # get photon budget and photobleaching-time
            photons = (tau[i]/tau0)*photon0

            N = 10000 #int((end - start)/dt)

            time  = numpy.array([j*dt + start for j in range(N)])
            state = numpy.zeros(shape=(N))

#           #### Photoactivation : overall activation yeild
#           p = self.photoactivation_activation_yield
#           q = self.photoactivation_frac_preactivation
#
#           r = numpy.random.uniform(0, 1, N/F*f)
#           s = (r < p).astype('int')
#
#           index = numpy.abs(s - 1).argmin()
#           t0 = (index/f*F + numpy.remainder(index, f))*dt + start

            t0 = start
            t1 = t0 + tau[i]

            N0 = (numpy.abs(time - t0)).argmin()
            N1 = (numpy.abs(time - t1)).argmin()
            state[N0:N1] = numpy.ones(shape=(N1-N0))

            if (self.photoblinking_switch == True) :

                # ON-state
                t_on  = numpy.random.choice(tau_on,  100, p=p_on)
                # OFF-state
                t_off = numpy.random.choice(tau_off, 100, p=p_off)

                k = (numpy.abs(numpy.cumsum(t_on) - tau[i])).argmin()

                # merge t_on/off arrays
                t_blink = numpy.array([t_on[0:k], t_off[0:k]])
                t = numpy.reshape(t_blink, 2*k, order='F')

                i_on  = N0

                for j in range(k) :

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
