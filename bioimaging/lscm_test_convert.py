import sys
import math
import copy

import pylab
import numpy

from Image import fromarray
from scipy.misc import imread, toimage
from scipy.special import j0, gamma

cmin = 0
cmax = 400

def prob_analog(y, alpha) :

    # get average gain
    A  = 1e+6

    # get dynodes stages
    nu = 11

    B = 0.5*(A - 1)/(A**(1.0/nu) - 1)
    c = numpy.exp(alpha*(numpy.exp(-A/B) - 1))

    m_y = alpha*A
    m_x = m_y/(1 - c)

    s2_y = alpha*(A**2 + 2*A*B)
    s2_x = s2_y/(1 - c) - c*m_x**2

    #if (y < 10*A) :
    # Rayleigh approximation
    #s2 = (2.0/numpy.pi)*m_x**2
    #prob = y/s2*numpy.exp(-0.5*y**2/s2)

    # probability distribution
    prob = numpy.zeros(shape=(len(y)))

    # get index
    k = (numpy.abs(10*A - y)).argmin()

    # Gamma approximation
    k_1 = m_x
    k_2 = (m_y**2 + s2_y)/(1 - c)

    a = 1/(k_1*(k_2/k_1**2 - 1))
    b = a*k_1

    prob[0:k] = a/gamma(b)*(a*y[0:k])**(b-1)*numpy.exp(-a*y[0:k])

    if (k < len(y)) :
        # Truncated Gaussian approximation
        Q = 0
        beta0 = m_x/numpy.sqrt(s2_x)
        beta  = beta0
        delta = 0.1*beta0

        while (beta < 11*beta0) :
            Q += numpy.exp(-0.5*beta**2)/numpy.sqrt(2*numpy.pi)*delta
            beta += delta

        prob[k:] = numpy.exp(-0.5*(y[k:] - m_x)**2/s2_x)/(numpy.sqrt(2*numpy.pi*s2_x)*(1 - Q))

    return prob




def convert(file_in, file_out, index=None) :

    i = 0

    max_count = 0
    image_signal = numpy.zeros([100,100])

    while (True) :
        try :
            input_image  = numpy.load(file_in + '/image_%07d.npy' % (i))
        except Exception :
            break

        output_image = file_out + '/image_%07d.png' % (i)
        #output_image = file_out + '/image_%07d.png' % (i/26)

        # data for tirfm
        image_expected = input_image[512-50:512+50,512-50:512+50,0]

#           amax = numpy.amax(image_expected)
#           amin = numpy.amin(image_expected)
#
#           if (max_count < amax) :
#               max_count = amax
#
#           print image_expected
#           print i, amax, amin

        #Background = 0
        Background = 0

        for x in range(100) :
            for y in range(100) :
                # expectation
                Exp = image_expected[x,y] + Background

                # get EM gain
                G = 1e+6

                # signal array
                if (Exp > 1) : sig = numpy.sqrt(Exp)
                else : sig = 1

                s_min = int(G*(Exp - 10*sig))
                s_max = int(G*(Exp + 10*sig))

                if (s_min < 0) : s_min = 0

                delta = (s_max - s_min)/1000.

                s = numpy.array([k*delta+s_min for k in range(1000)])

                # probability density fuction
                p_signal = prob_analog(s, Exp)
                p_ssum = p_signal.sum()

                # get signal (photoelectrons)
                signal = numpy.random.choice(s, None, p=p_signal/p_ssum)

#                   # get detector noise (photoelectrons)
#                   Nr = 0.001*G
#                   noise = numpy.random.normal(Nr, numpy.sqrt(Nr), None)

                image_signal[x,y] = signal/1e+4
                #image_signal[x,y] = signal + noise

        amax = numpy.amax(image_signal)
        amin = numpy.amin(image_signal)

        if (max_count < amax) :
            max_count = amax

        print(image_signal)
        print(i, amax, amin)

        image_signal2 = (image_signal - amin)*(4336-1424)/(amax-amin) + 1424

        # 16-bit data format
        image_signal.astype('uint16')
        toimage(image_signal2, high=4336, low=1424, mode='I').save(output_image)

        # 8-bit data format (for making movie)
        #toimage(image_signal, cmin=cmin, cmax=cmax).save(output_image)

        i += 1
        break


    print('Max count : ', max_count, 'ADC')



if __name__=='__main__':

    file_in  = '/home/masaki/bioimaging_4public/images_lscm_test_010uW'
    #file_in  = '/home/masaki/bioimaging_4public/images_lscm_test_z00_100uW'
    file_out = '/home/masaki/bioimaging_4public/images_png'

    convert(file_in, file_out)
