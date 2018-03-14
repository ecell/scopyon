import sys
import math
import copy

import pylab
import numpy


def convert(file_in) :

	i = 0
	result=numpy.zeros(shape=(2e+5,3))

	while (True) :
	    try :
		npy_file = numpy.load(file_in + '/image_%07d.npy' % (i))
		result[i,:] = numpy.array(npy_file)
		#print i, npy_file
		#result = numpy.append(result, npy_file)

	    except Exception :
		break

	    i += 1
	    if (i > 2e+5) : break


        sav = numpy.save(file_in + '/image_all.npy', result)



if __name__=='__main__':

	file_in  = '/home/masaki/bioimaging_4public/images_fcs_2StMD_A200'
	#file_in  = '/home/masaki/bioimaging_4public/images_fcs_D010_A200'

	convert(file_in)

