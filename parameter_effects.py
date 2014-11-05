
import numpy

#-----------------------------
# Linear conversion of input to output intensity
#-----------------------------
conversion_ratio = 1e-6

#-----------------------------
# Background
#-----------------------------
background_switch = False
background_mean = 0
background_width = 0

#-----------------------------
# Detectors Crosstalk
#-----------------------------
detector_crosstalk_switch = False
detector_crosstalk_width  = 0.0

#-----------------------------
# Photobleaching
#-----------------------------
bleaching_switch = False
bleaching_rate = 0.00

bleaching_state = []

#-----------------------------
# Photoblinking
#-----------------------------
blinking_switch = False
blinking_prob0 = 1.00

blinking_alpha_on  = 0.00
blinking_alpha_off = 0.00

blinking_state  = []
blinking_period = []


