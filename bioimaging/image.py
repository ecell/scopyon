import os.path
import numpy


def convert_8bit(data, cmin=None, cmax=None, low=None, high=None):
    """Same as scipy.misc.bytescale"""
    data = numpy.asarray(data)
    cmin = data.min() if cmin is None else cmin
    cmax = data.max() if cmax is None else cmax
    low = 0 if low is None else low
    high = 255 if high is None else high

    cscale = cmax - cmin
    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    bytedata = (bytedata.clip(low, high) + 0.5).astype(numpy.uint8)
    return bytedata

def save_as_image(filename, data, cmap=None, low=None, high=None):
    data = numpy.asarray(data)
    low = data.min() if low is None else low
    high = data.max() if high is None else high

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    cmap = cmap or cm.gray
    plt.imsave(filename, data, cmap=cmap, vmin=low, vmax=high)

def convert_npy_to_8bit_image(filename, output=None, cmap=None, cmin=None, cmax=None, low=None, high=None):
    low = 0 if low is None else low
    high = 255 if high is None else high

    output = output or '{}.png'.format(os.path.splitext(filename)[0])

    data = numpy.load(filename)
    data = data[: , : , 1]

    bytedata = convert_8bit(data, cmin, cmax, low, high)
    save_as_image(output, bytedata, cmap, low, high)
