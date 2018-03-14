import os
import sys
import math

#import pylab
import numpy
#import scipy
#from scipy.misc import comb
from scipy.special import binom

# photo-bleaching effect
a_bleach  = 0.74
t0_bleach = 6.80 # sec

# photo-blinking effect (if a_on > a_off) [photo-brightening effect (if a_on < a_off)]
# ON-state
a_on  = 1.58
t0_on = 1.30 # sec

# OFF-state
a_off  = 1.48
t0_off = 1.30 # sec

# exposure time
dt = 30e-3 # sec

def prob_levy(t, t0, a) :

    prob = (a/t0)*(t0/t)**(1+a)

    return prob


def convert(start, end) :

    ##### set tau-array for probability density fuction
    NNN = int(1e+7)
    tau_on  = numpy.array([i*dt+t0_on  for i in range(NNN)])
    tau_off = numpy.array([i*dt+t0_off for i in range(NNN)])
    tau_bleach = numpy.array([i*dt+t0_bleach for i in range(NNN)])

    ##### set photo-blinking effects
    NN = int(end/dt/2)

    # ON-state
    prob_on = prob_levy(tau_on, t0_on, a_on)
    prob_on_sum = prob_on.sum()
    p_on = prob_on/prob_on_sum

    t_on = numpy.random.choice(tau_on, NN, p=p_on)

    # OFF-state
    prob_off = prob_levy(tau_off, t0_off, a_off)
    prob_off_sum = prob_off.sum()
    p_off = prob_off/prob_off_sum

    t_off = numpy.random.choice(tau_off, NN, p=p_off)

    # merge t_on/off arrays
    a = numpy.array([t_on, t_off])
    t_onoff = numpy.reshape(a, 2*NN, order='F')
    t = numpy.cumsum(t_onoff)

    # set state array
    s_on  = numpy.ones(shape=(1, NN))
    s_off = numpy.zeros(shape=(1, NN))
    c = numpy.array([s_on, s_off])
    s = numpy.reshape(c, 2*NN, order='F')

    ##### set photo-bleaching effect
    prob_bleach = prob_levy(tau_bleach, t0_bleach, a_bleach)
    prob_bleach_sum = prob_bleach.sum()
    p_bleach = prob_bleach/prob_bleach_sum

    t_bleach = numpy.random.choice(tau_bleach, None, p=p_bleach)

    #####
    # Active states per detection time
    t_life = 30e-9
    N_act0 = int(dt/t_life)
    N_act  = N_act0

    # set active states (All)
    N_tot = 0

    # initial
    Na = 6.022e+23
    hc = 2.00e-25
    wl = 532e-9
    E = hc/wl
    abs_coeff = 1e+5 # 1/(cm M)
    xsec = numpy.log(10)*0.1*abs_coeff/Na # m2
    QY = 1.0
    P = 20 # W/cm2

    r = 20e-9
    volume = 4.0/3.0*numpy.pi*r**3
    depth  = 2.0*r

    # the number of absorbed photons
    N_abs = (xsec*1e+4)*(P/E)*dt

    # Beer-Lambert law : A = log(I0/I) = coef * concentration * path length
    A = (abs_coeff*0.1/Na)*(1/volume)*depth

    # the number of emitted photons
    N_emit0 = QY*N_abs*(1 - 10**(-A))

    # set permanent dark state (PDS)
    N_pds = 0

    time = start

    while (time < end) :
        N_on = N_off = 0
        N_emit = 0

        # get current index of blinking-state array
        i = (numpy.abs(t - time)).argmin()

        # Photo-blinking (ON-state)
        if (s[i] == 1) :
            N_on = N_act

            # Fraction of active-molecules
            f = N_act/N_act0

            # The number of emitted photons
            N_emit = N_emit0 * f

        # Photo-blinking (OFF-state)
        else :
            N_off = N_act

            # Photo-bleaching (to permanent dark state)
            #dN = N_off*(1/t_pulse + 1/t_bleach)*dt
            dN = N_off*(1/t_bleach)*dt
            N_pds = N_pds + dN
            N_off = N_off - dN

        # the number of active states at a given time
        N_act = N_on + N_off

        # the number of active states (Total)
        N_tot = N_tot + N_act

        print(time, N_on, N_off, N_pds, N_tot, N_emit)

        time += dt


if __name__=='__main__':

    start = float(sys.argv[1])
    end   = float(sys.argv[2])

    convert(start, end)
