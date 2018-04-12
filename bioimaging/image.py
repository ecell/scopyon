import numpy

def convert_8bit(data, cmin=None, cmax=None, low=0, high=255):
    """Same as scipy.misc.bytescale"""
    data = numpy.asarray(data)
    cmin = cmin or data.min()
    cmax = cmax or data.max()

    cscale = cmax - cmin
    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    bytedata = (bytedata.clip(low, high) + 0.5).astype(numpy.uint8)
    return bytedata

def save_as_image(filename, data, cmap=None, low=None, high=None):
    data = numpy.asarray(data)
    low = low or data.min()
    high = high or data.max()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    cmap = cmap or cm.gray
    plt.imsave(output_filename, bytedata, cmap=cmap, vmin=low, vmax=high)

def convert_npy_to_8bit_image(filename, output=None, cmap=None, cmin=None, cmax=None, low=None, high=None):
    output = output or '{}.png'.format(os.path.splitext(filename)[0])

    data = numpy.load(filename)
    data = data[: , : , 1]

    bytedata = convert_8bit(data, cmin, cmax, low, high)
    save_as_image(output, bytedata, cmap, low, high)
