import sys
import math
import copy

import pylab
import numpy

from Image import fromarray
from scipy.misc import imread, toimage

cmin = 0
cmax = 2**8 - 1 

def convert(file_in, file_out, index=None) :

	i = 0
	max_count = 0

	while (True) :
	    try :
		input_image  = numpy.load(file_in + '/image_%07d.npy' % (i))
	    except Exception :
		break

	    output_image = file_out + '/image_%07d.png' % (i)
	    #output_image = file_out + '/image_%07d.png' % (i/26)

	    # data for tirfm
	    #image_array = input_image[256-25:256+25,256-25:256+26,1]
	    #image_array = input_image[256-76:256+76,256-78:256+78,1]
	    #image_array = input_image[300-50:300+50,300-50:300+50,1]
	    #image_array = input_image[512-45:512+45,512-45:512+45,1]
	    image_array = input_image[:,:,1]

	    #image_exp += numpy.array(image_array)

	    amax = numpy.amax(image_array)
	    amin = numpy.amin(image_array)

	    if (max_count < amax) :
		max_count = amax

	    #print i/26, amax, amin
	    print i, amax, amin

	    # 16-bit data format
	    #image_array.astype('uint16')
	    #toimage(image_array, low=cmin, high=cmax, mode='I').save(output_image)

	    # 8-bit data format (for making movie)
	    toimage(image_array, cmin=cmin, cmax=cmax).save(output_image)

	    #i += 26
	    i += 1

	print 'Max count : ', max_count, 'ADC'



if __name__=='__main__':

	file_in  = '/home/masaki/microscopy/images'
	file_out = '/home/masaki/microscopy/images_png'

	convert(file_in, file_out)

