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

cmin = 1900
cmax = 3600

def convert(file_in, file_out) :

    i = 0
    max_count = 0

    while (True) :
        try :
            input_image  = numpy.load(file_in + '/image_%07d.npy' % (i))
            #input_image  = numpy.load(file_in + '/' + filename[i])
        except Exception :
            break

        output_image = file_out + '/image_%07d.png' % (i)
        #output_image = file_out + '/image_%07d_%02ddeg.png' % (i, angle)
        #output_filename = filename[i].split('.npy')
        #output_image = file_out + '/%s.png' % (output_filename[0])

        # data for tirfm
        #image_array = input_image[256-200:256+200,256-200:256+200,1]
        #image_array = input_image[256-76:256+76,256-78:256+78,1]
        #image_array = input_image[300-25:300+25,300-100:300-100+50,1]
        #image_array = input_image[200-100:200+100,200-100:200+100,1]
        #image_array = input_image[300-200:300+200,300-200:300+200,1]
        #image_array = input_image[256-100:256+100,256-100:256+100,1]
        #image_array = input_image[300-50:300+50,300-50:300+50,1]
        image_array = input_image[:,:,1]

        #image_exp += numpy.array(image_array)

        amax = numpy.amax(image_array)
        amin = numpy.amin(image_array)

        if (max_count < amax) :
            max_count = amax

        print(i, amax, amin)

        # 16-bit data format
        #image_array.astype('uint16')
        #toimage(image_array, high=11000, low=100, mode='I').save(output_image)

        # 8-bit data format (for making movie)
        toimage(image_array, cmin=cmin, cmax=cmax).save(output_image)

        i += 1

    print('Max count : ', max_count, 'ADC')



if __name__=='__main__':

    file_in  = '/home/masaki/bioimaging_4public/data_hiroshima_for_calibration/numpys/numpys_Test_00'
    file_out = '/home/masaki/bioimaging_4public/data_hiroshima_for_calibration/images_png'

    convert(file_in, file_out)
