import os
import sys
import math
import copy
import csv
import subprocess
import pylab
import numpy

from Image import fromarray
from scipy.misc import imread, toimage
import shutil

#cmin = 100
#cmax = 1000

def convert(file_in, file_out, cmin, cmax) :

	i = 0
	max_count = 0

	while (True) :
	    try :
		input_image  = numpy.load(file_in + '/image_%07d.npy' % (i))
		#input_image  = numpy.load(file_in + '/' + filename[i])
	    except Exception :
		break

	    output_image = file_out + '/image_%07d.png' % (i)
	    #output_filename = filename[i].split('.npy')
	    #output_image = file_out + '/%s.png' % (output_filename[0])

	    # data for tirfm
	    #image_array = input_image[256-25:256+25,256-25:256+26,1]
	    #image_array = input_image[256-76:256+76,256-78:256+78,1]
	    #image_array = input_image[300-25:300+25,300-100:300-100+50,1]
	    #image_array = input_image[200-100:200+100,200-100:200+100,1]
	    #image_array = input_image[300-200:300+200,300-200:300+200,1]
	    #image_array = input_image[512-100:512+100,512-100:512+100,1]
	    image_array = input_image[300-50:300+50,300-50:300+50,1]
	    #image_array = input_image[:,:,1]

	    #image_exp += numpy.array(image_array)

	    amax = numpy.amax(image_array)
	    amin = numpy.amin(image_array)

	    if (max_count < amax) :
		max_count = amax

	    #print i, amax, amin

	    # 16-bit data format
	    #image_array.astype('uint16')
	    #toimage(image_array, high=11000, low=100, mode='I').save(output_image)

	    # 8-bit data format (for making movie)
	    toimage(image_array, cmin=cmin, cmax=cmax).save(output_image)

	    i += 1

	print 'Max count : ', max_count, 'ADC'



if __name__=='__main__':

	#file_in  = '/home/masaki/bioimaging_4public/data_moriga_01/numpys/model_001A/index_099'
	#file_in  = '/home/masaki/bioimaging_4public/numpys_palm_aggr00_bleach'
	#file_in  = '/home/masaki/bioimaging_4public/numpys_palm_aggr05_blink'
	#file_in  = '/home/masaki/bioimaging_4public/numpys_lscm_019656A'
	#file_in  = '/home/masaki/bioimaging_4public/numpys_lscm_019656A_bleach'
	#file_in  = '/home/masaki/bioimaging_4public/numpys_flip_080000A_bleach'
	#file_in  = '/home/masaki/bioimaging_4public/numpys_frap_019656A_bleach'
	#file_in  = '/home/masaki/bioimaging_4public/numpys_frap_000100A_bleach'
	#file_out = '/home/masaki/bioimaging_4public/images_png'

	#convert(file_in, file_out, 0, 20)

	#nm = int(sys.argv[1])

	I0 = [40, 50, 60 ,70, 80, 90]
	Bg = [7, 4, 2, 0]

	cmin, cmax = 100, 400

	file_out = '/home/masaki/bioimaging_4public/data_moriga_02/images/coordinate_002A_extfunc.csv'

	for nm in range(20, 380, 20) :
	#for i in range(len(I0)) :
	   #for j in range(len(Bg)) :

	    file_in  = '/home/masaki/wrk/spatiocyte/models/data_moriga/model_002A_%03dnm/pt-000000000.0.csv' % (nm)
#	        file_out = '/home/masaki/bioimaging_4public/data_moriga_02/images/model_003A_%03dnm_%02dI0_%02dBg' % (nm, I0[2], Bg[0])

	    j = 0
	    for row in csv.reader(open(file_in, 'r')) :

	        line  = str(nm) + ','
	        line += str(46) + ','
	        line += str(7) + ','
		line += row[1] + ','
		line += row[2] + ','
		line += row[3] + ','
		line += row[4] + ','
		line += '"' + row[5] + '",'
		line += '"' + row[6] + '"\n'

	    	with open(file_out, 'a') as output :
		    output.write(line)

		j += 1

#	    print file_in
#	    print file_out
#
#	    shutil.copy2(file_in, file_out)
#
#	        if not os.path.exists(file_out):
#		    os.makedirs(file_out)
#
#	        convert(file_in, file_out, cmin, cmax)

