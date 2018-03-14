import sys
import math
import copy

import pylab
import numpy
from scipy import optimize

from scipy.misc import imread, toimage

cmin = 100
cmax = 600

def convert(file_in, file_out, index=None) :

	i = 1

	while (True) :
	    try :
		#input_image  = numpy.load(file_in + '/image_%07d.npy' % (i))
	        input_image_0 = (imread(file_in + '/image_%07d.tif' % (i-1))).astype('int')
	        input_image_1 = (imread(file_in + '/image_%07d.tif' % (i))).astype('int')
	    except Exception :
		break

	    output_image = file_out + '/image_%07d.png' % (i-1)

	    # data for tirfm
	    diff_image = input_image_0 - input_image_1
	    hist_image_1 = input_image_1.reshape((512*512))

	    length = len(hist_image_1)

	    avg = hist_image_1.sum()/length
	    dev = numpy.sqrt(avg)

	    check_diff = (diff_image > 0).astype('int')
	    image0 = numpy.abs(diff_image*check_diff)

	    #######
	    length = 512*512
	    n = 16
	    N = length/(n*n)
	    image1 = image0.reshape((N,n,n))

	    avg  = image1.sum(axis=1).sum(axis=1)/float(n*n)
	    check1 = (avg > 0).astype('int')

	    for j in range(N) :
	        image1[j] = numpy.abs(image1[j]*check1[j])

	    image = image1.reshape((512,512))
	    #image = input_image_0.reshape((512,512))

	    amin = numpy.amin(image)
	    amax = numpy.amax(image)

	    print amin, amax

	    # 16-bit data format
	    #image_array.astype('uint16')
	    #toimage(image_array, high=cmax, low=cmin, mode='I').save(output_image)

	    # 8-bit data format (for making movie)
	    toimage(image, cmin=amin, cmax=500).save(output_image)

	    i += 1



if __name__=='__main__':

	file_in  = '/home/masaki/bioimaging_4public/Data_fromMoriga_08-05-2015/images_0805/images_tif_220mW'
	file_out = '/home/masaki/bioimaging_4public/images_png'

	convert(file_in, file_out)

