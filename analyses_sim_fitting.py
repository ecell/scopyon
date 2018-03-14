import os
import sys
import math
import copy
import subprocess

import pylab
import numpy
from scipy import optimize

from scipy.misc import imread, toimage

def gaussian(A, x0, y0, width_x, width_y) :
	"""Returns a gaussian function with the given parameters"""
	width_x = float(width_x)
	width_y = float(width_y)

	return lambda x,y: A*numpy.exp(-(((x0-x)/width_x)**2+((y0-y)/width_y)**2)/2)


def moments(data) :
	"""Returns (A, x, y, width_x, width_y)
	the gaussian parameters of a 2D distribution by calculating its
	moments """
	total = data.sum()

	X, Y = numpy.indices(data.shape)
	x = (X*data).sum()/total
	y = (Y*data).sum()/total

	col = data[:, int(y)]
	width_x = numpy.sqrt(abs((numpy.arange(col.size)-y)**2*col).sum()/col.sum())

	row = data[int(x), :]
	width_y = numpy.sqrt(abs((numpy.arange(row.size)-x)**2*row).sum()/row.sum())

	height = data.max()

	return height, x, y, width_x, width_y



def convert(file_in, file_out) :

	Np = 6.5e-6/89/1e-9
	x_nm, y_nm = 2.5057e-05/1e-9, 1.432e-05/1e-9
	x_px, y_px = 94 + x_nm/Np - 44, 94 + y_nm/Np - 44

	proc = subprocess.Popen(["ls %s" % (file_in)], stdout=subprocess.PIPE, shell=True)
	(out, err) = proc.communicate()
	filename = out.split('\n')

	for i in range(len(filename)) :

	    try :
		input_image = numpy.load(file_in + '/' + filename[i]).astype('int')
	    except Exception :
		break

	    array_image = input_image[300-256:300+256,300-256:300+256,1]

	    # get intensity
	    temp0 = filename[i].split('_')
	    temp1 = temp0[1].split('.npy')
	    intensity = float(temp1[0])

	    # histogram
	    hist_image = array_image.reshape((512*512))

	    length = len(hist_image)

	    B_avg = hist_image.sum()/float(length)
	    Bdev2 = (hist_image - B_avg)**2
	    B_dev = numpy.sqrt(Bdev2.sum()/length)

	    x, y = 394, 247
	    image1 = array_image[y-8:y+8,x-8:x+8] - B_avg
	    check1 = (image1 > 0).astype('int')
	    image0 = numpy.abs(check1*image1)
	    I_sum = image0.sum()

	    params = moments(image0)
	    errorfunction = lambda p : numpy.ravel(gaussian(*p)(*numpy.indices(image0.shape)) - image0)
	    p, success = optimize.leastsq(errorfunction, params)

	    print intensity, x_px-(x-8), y_px-(y-8), I_sum, p[2], p[1], B_avg, B_dev
	    #print intensity, x_px-(x-8), y_px-(y-8), p, success, B_avg, B_dev


if __name__=='__main__':

	file_in = '/home/masaki/bioimaging_4public/numpys_4moriga_210mW'
	file_out = '/home/masaki/bioimaging_4public/images_png'

	convert(file_in, file_out)

