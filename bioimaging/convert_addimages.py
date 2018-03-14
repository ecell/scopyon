import os
import sys
import math
import copy
import csv
import subprocess
import pylab
import numpy

#from matplotlib import image
from Image import fromarray
from scipy.misc import imread, toimage
import shutil

cmin = 1900
cmax = 4000

def convert(file_in0, file_in1, file_out) :

    i = 20000

    while (True) :
        try :
            #input_image0  = imread(file_in0 + '/image_%07d.png' % (i+1))
            #input_image1  = imread(file_in1 + '/image_%07d.png' % (i))
            input_image0  = numpy.load(file_in0 + '/image_%07d.npy' % (i))
            input_image1  = numpy.load(file_in1 + '/image_%07d.npy' % (i))

        except Exception :
            break

        output_image = file_out + '/image_%07d.png' % (i-20000)

        # data for tirfm
        #image_array = numpy.zeros(shape=(512,2*512))
        #image_array[:,0:512] = input_image0[:,:,1]
        #image_array[:,512:2*512] =  input_image1[:,:,1]
        image_array = numpy.zeros(shape=(256,2*256))
        image_array[:,0:256] = input_image0[256-128:256+128,306-128:306+128,1]
        image_array[:,256:2*256] =  input_image1[256-128:256+128,306-128:306+128,1]

        amax = numpy.amax(image_array)
        amin = numpy.amin(image_array)

        #print i, amax, amin

        #photons0 = 5.82*(numpy.sum(input_image0[:,:,1])-2000)/300/0.92
        #photons1 = 5.82*(numpy.sum(input_image1[:,:,1])-2000)/300/0.92
        photons0 = 5.82*(numpy.sum(input_image0[256-128:256+128,306-128:306+128,1])-2000)/300/0.92
        photons1 = 5.82*(numpy.sum(input_image1[256-128:256+128,306-128:306+128,1])-2000)/300/0.92

        print(i*0.150, photons0, photons1)

        # 16-bit data format
        #image_array.astype('uint16')
        #toimage(image_array, high=11000, low=100, mode='I').save(output_image)

        # 8-bit data format (for making movie)
        #toimage(image_array, cmin=cmin, cmax=cmax).save(output_image)

        i += 1




if __name__=='__main__':

    #file_in0 = '/home/masaki/bioimaging_4public/images_experiment'
    #file_in1 = '/home/masaki/bioimaging_4public/images_oligomer_model'
    #file_in0 = '/home/masaki/bioimaging_4public/numpys_hiroshima_TEST_on'
    #file_in1 = '/home/masaki/bioimaging_4public/numpys_hiroshima_TEST_off'
    file_in0 = '/home/masaki/bioimaging_4public/numpys_hiroshima_Olg_001nM_on'
    file_in1 = '/home/masaki/bioimaging_4public/numpys_hiroshima_Olg_001nM_off'
    file_out = '/home/masaki/bioimaging_4public/images_png'

    convert(file_in0, file_in1, file_out)
