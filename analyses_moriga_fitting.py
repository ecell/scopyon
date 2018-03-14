import sys
import math
import copy

import pylab
import numpy
from scipy import optimize

from scipy.misc import imread, toimage

#coord = [(0, 384,228),(0,283,351),(0,238,406),(1,219,265),(1,313,311),(1,360,280),(1,239,407),(3,275,261),(3,291,286),(3,275,362),(4,251,178),(5,247,179),(5,208,262),(5,246,178),(6,259,186),(6,189,260),(8,218,276),(8,243,298),(9,200,254),(9,276,235)]
#coord = [(0,286,231),(0,327,313),(3,342,275),(3,355,288),(4,359,293),(5,236,180),(5,325,246),(5,368,263),(7,346,290),(7,358,292),(8,242,282),(8,208,211)]
#coord = [(0,368,291),(3,356,210),(3,351,225),(3,308,229),(4,337,283),(4,234,187),(4,354,211),(5,221,346),(7,210,272),(7,150,372),(8,334,266),(8,361,286),(8,226,315),(9,228,306)]
#coord = [(0,241,192),(0,205,213),(1,366,300),(2,369,238),(2,252,315),(5,278,286),(5,268,274),(6,179,246),(7,209,269),(7,303,306),(7,479,427)]
#coord = [(0,287,365),(1,354,267),(3,231,198),(5,297,342),(5,478,268),(6,323,241),(6,279,144),(6,244,266),(6,357,304),(8,255,196),(9,248,186),(9,243,177),(9,114,332)]
#coord = [(1,283,288),(1,333,373),(1,356,421),(2,214,340),(2,325,310),(2,332,230),(2,323,249),(3,367,256),(3,387,116),(4,213,417),(5,329,281),(9,260,257)]
coord = [(0,334,295),(0,368,472),(0,387,23),(2,249,186),(2,407,148),(3,325,317),(4,336,505),(6,275,281),(9,329,438),(9,235,369)]


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



def convert(file_in, file_out, index=None) :

	i = 1

	#image = numpy.zeros(shape=(512,512))
	signal = []
	background = []
	error  = []

	while (True) :
	    try :
		#input_image  = numpy.load(file_in + '/image_%07d.npy' % (i))
	        input_image = (imread(file_in + '/image_%07d.tif' % (i-1))).astype('int')
	    except Exception :
		break

	    image0 = numpy.array(input_image)

	    for j in range(len(coord)) :

	      if (i-1 == coord[j][0]) :

		x, y = coord[j][1], coord[j][2]

		image1 = input_image[y-8:y+8,x-8:x+8]
		image0[y-8:y+8,x-8:x+8] = image0[y-8:y+8,x-8:x+8] - image1

		I_all = float(image1.sum())
		signal.append(I_all)

#	        params = moments(image0)
#	        errorfunction = lambda p : numpy.ravel(gaussian(*p)(*numpy.indices(image0.shape)) - image0)
#	        p, success = optimize.leastsq(errorfunction, params)

	    I_bg = []
	    array0 = image0.reshape((512*512))

	    for k in range(len(array0)) :

		if (array0[k] > 0) :
		    I_bg.append(float(array0[k]))

	    I_bg = numpy.array(I_bg)

	    length = len(I_bg)

	    b_avg = I_bg.sum()/length
	    bdev2 = (I_bg - b_avg)**2
	    dev_b = numpy.sqrt(bdev2.sum()/length)

	    background.append(b_avg)
	    error.append(dev_b)

	    # 16-bit data format
	    #image_array.astype('uint16')
	    #toimage(image_array, high=cmax, low=cmin, mode='I').save(output_image)

	    # 8-bit data format (for making movie)
	    #toimage(image, cmin=amin, cmax=410).save(output_image)

	    i += 1

	# background
	background = numpy.array(background)
	length = len(background)
	b_avg = background.sum()/length

	error = numpy.array(error)
	length = len(error)
	err_b = error.sum()/length

	# signal
	signal = numpy.array(signal - 16*16*b_avg)

	length = len(signal)
	s_avg = signal.sum()/length
	dev2  = (signal - s_avg)**2
	err_s = numpy.sqrt(dev2.sum()/length)

	print 210, s_avg, err_s, b_avg, err_b

	# 8-bit data format (for making movie)
	#toimage(image, cmin=100, cmax=1000).save(file_out + '/image_summed.png')


if __name__=='__main__':

	file_in = '/home/masaki/bioimaging_4public/Data_fromMoriga_08-05-2015/images_0805/images_tif_210mW'
	file_out = '/home/masaki/bioimaging_4public/images_png'

	convert(file_in, file_out)



