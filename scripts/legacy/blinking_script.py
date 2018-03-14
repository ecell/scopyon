import sys
import os
import numpy

from palm_handler   import PALMConfigs, PALMVisualizer
from effects_handler import PhysicalEffects

def test_palm(t0, t1) :

#       # create PALM imaging
#       palm = PALMConfigs()
#       palm.set_ExcitationSource(source_type='LASER', wave_length=561, flux_density=30, angle=65.7)
#       palm.set_ActivationSource(source_type='LASER', wave_length=504, flux_density=20, angle=65.7, frame_time=0.5, F_bleach=50)
#       palm.set_Fluorophore(fluorophore_type='EGFP')
##      palm.set_DichroicMirror('FF562-Di03-25x36')
##      palm.set_EmissionFilter('FF01-593_40-25')
#       palm.set_Magnification(Mag=100)
#
#       # Detector : CMOS Camera
#       palm.set_Detector(detector='CMOS', image_size=(600,600), pixel_length=6.5e-6, \
#                       focal_point=(0.0,0.5,0.5), exposure_time=0.5, QE=0.73)
#       palm.set_ADConverter(bit=16, offset=100, fullwell=30000)
#
##      # Detector : EMCCD Camera
##      palm.set_Detector(detector='EMCCD', image_size=(512,512), pixel_length=16e-6, \
##                      focal_point=(0.0,0.5,0.5), exposure_time=30e-3, QE=0.92, readout_noise=50, emgain=300)
##      palm.set_ADConverter(bit=14, offset=2000, fullwell=800000)
##
##      # Output data
##      palm.set_OutputData(image_file_dir='./numpys_palm_01000A')
#
#       # Input data
#       palm.reset_InputData('/home/masaki/ecell3/latest/data/csv/beads_palm_10000A', start=t0, end=t1, observable="A")
#       #palm.set_InputData('/home/masaki/bioimaging_4public/data_palm_01000A', start=t0, end=t1, observable="A")

    # create physical effects
    physics = PhysicalEffects()
    physics.set_background(mean=0)
    physics.set_fluorescence(photon_budget=200, quantum_yield=0.61, abs_coefficient=83400, alpha=0.10)
    physics.set_photoactivation(turn_on_ratio=1000, activation_yield=1.0, frac_preactivation=0.00)
    physics.set_photoblinking(t0_on=0.30, a_on=0.58, t0_off=10e-6, a_off=0.48)

    dt = 0.2
    n_emit0 = 1
    physics.set_photophysics(t0, t1, dt, 1, 1, n_emit0, 1)
    budget = physics.fluorescence_budget[0]
    state  = physics.fluorescence_state[0]

    print('#', budget, 'photons', (budget/n_emit0)*dt, 'sec')

    for i in range(len(state)) :
        print(dt*i, state[i], budget)

        if (state[i] > 0) :
            budget = budget - n_emit0


if __name__ == "__main__":

    t0 = float(sys.argv[1])
    t1 = float(sys.argv[2])

    test_palm(t0, t1)
