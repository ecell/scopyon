import sys
import math
import copy

import scipy
import numpy


def get_auto_corr(I0, It) :

    length = len(It[:,2])
    mu = It[:,2].sum()/length
    s2 = (It[:,2] - mu)**2
    s  = numpy.sqrt(s2.sum()/(length-1))

    cor = (I0[:,2] - mu)*(It[:,2] - mu)
    avg = cor.sum()/len(cor)

    C_auto = avg/(mu*mu)

    return C_auto


if __name__=='__main__':

    file_in = '/home/masaki/bioimaging_4public/images_fcs_2StMD_A200'
    #file_in = '/home/masaki/bioimaging_4public/images_fcs_D010_A100'
    #file_in = '/home/masaki/bioimaging_4public/images_fcs_D010_A200'
    #file_in = '/home/masaki/bioimaging_4public/images_fcs_D100_A100'
    #file_in = '/home/masaki/bioimaging_4public/images_fcs_D100_A200'

    array = numpy.load(file_in + '/image_all.npy')

    N = len(array)/2

    for i in range(N) :

        I0 = array[0:N]
        It = array[i:i+N]

        C_auto= get_auto_corr(I0, It)

        print(It[i,0], It[i,2], C_auto)
