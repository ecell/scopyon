import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Avogadro constant
# AVOGADROS_NUMBER = 6.022140857e+23
AVOGADROS_NUMBER = (1.0 * ureg.avogadro_number).to('dimensionless').magnitude
N_A = AVOGADROS_NUMBER

# (plank const) * (speed of light) [joules meter]
# hc = 6.62607015e-34 * 299792458 = 1.9864458571489289e-25
hc = (1.0 * ureg.planck_constant * ureg.speed_of_light).to(ureg.joule * ureg.meter).magnitude
