import numpy

avogadoros_number = 6.022e+23 # #/mol

#-----------------------------
# Shutter Open/Close period
#-----------------------------
shutter_switch = False
shutter_time_open  = 0
shutter_time_lapse = 0

#-----------------------------
# Fluorescence
#-----------------------------
background_switch = False
background_mean = 0
quantum_yield = 1.0
abs_coefficient = 1e+4

#-----------------------------
# Photoelectron Crosstalk
#-----------------------------
crosstalk_switch = False
crosstalk_width = 0

#-----------------------------
# Photo-bleaching
#-----------------------------
photobleaching_switch = False
photobleaching_tau0  = 6.8
photobleaching_alpha = 0.73

fluorescence_bleach = []
fluorescence_budget = []
fluorescence_state  = []

#-----------------------------
# Photo-activation
#-----------------------------
photoactivation_switch = False
photoactivation_turn_on_ratio = 1000
photoactivation_activation_yield = 0.1
photoactivation_frac_preactivation = 0.00

#-----------------------------
# Photo-blinking
#-----------------------------
photoblinking_switch = False
photoblinking_t0_on = 0.3
photoblinking_a_on  = 0.58
photoblinking_t0_off = 0.3
photoblinking_a_off  = 0.48

