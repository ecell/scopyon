import os
import sys
import shutil
import math
import numpy
import csv

def convert(file_in, file_out, start, end, nm, deg) :

	max_count = 0

	csv_input = file_in + '/pt-input.csv'
	shutil.copyfile(csv_input, file_out+ '/pt-input.csv')

	for i in range(start, end) :

	    time = i*0.1

	    output_csv = file_out + '/pt-%09d.0.csv' % (i)

	    y0, z0 = 2.50e-6, 2.50e-6
	    yyy = [y0]
	    zzz = [z0]
#	    r0 = 0.5*nm/numpy.cos(30*numpy.pi/180.)
#	    yyy = [y0+r0*numpy.cos(90*numpy.pi/180.), y0+r0*numpy.cos(210*numpy.pi/180.), y0+r0*numpy.cos(330*numpy.pi/180.)]
#	    zzz = [z0+r0*numpy.sin(90*numpy.pi/180.), z0+r0*numpy.sin(210*numpy.pi/180.), z0+r0*numpy.sin(330*numpy.pi/180.)]
#	    r0 = 0.5*nm
#	    yyy = [y0+r0*numpy.cos(45*numpy.pi/180.), y0+r0*numpy.cos(135*numpy.pi/180.), y0+r0*numpy.cos(225*numpy.pi/180.), y0+r0*numpy.cos(315*numpy.pi/180.)]
#	    zzz = [z0+r0*numpy.sin(45*numpy.pi/180.), z0+r0*numpy.sin(135*numpy.pi/180.), z0+r0*numpy.sin(225*numpy.pi/180.), z0+r0*numpy.sin(315*numpy.pi/180.)]

	    with open(output_csv, 'w') as output :
	        print(output_csv)

	    j = 0
	    for row in csv.reader(open(file_in+'/pt-000000000.0.csv', 'r')) :

	        line  = str(time) + ','
		line += row[1] + ','
		line += str(yyy[j]) + ','
		line += str(zzz[j]) + ','
		line += row[4] + ','
		line += '"' + row[5] + '",'
		line += '"' + row[6] + '"\n'

	    	with open(output_csv, 'a') as output :
		    output.write(line)

		j += 1


if __name__=='__main__':

	for nm in range(20, 380, 20) :
	#  for deg in range(0, 390, 30) :
	    deg = 0

	    file_in  = '/home/masaki/wrk/spatiocyte/models/data_moriga/model_001A/index_000'
	    #file_out = '/home/masaki/wrk/spatiocyte/models/data_moriga/model_002A_%03dnm_%03ddeg' % (nm, deg)
	    file_out = '/home/masaki/wrk/spatiocyte/models/data_moriga/model_001A_%03dnm' % (nm)

	    if not os.path.exists(file_out):
		os.makedirs(file_out)

	    convert(file_in, file_out, 0, 1000, nm*1e-9, deg)

