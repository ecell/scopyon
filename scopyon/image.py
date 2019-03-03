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

def save_image(filename, data, cmap=None, low=None, high=None, dpi=100):
    data = numpy.asarray(data)
    low = data.min() if low is None else low
    high = data.max() if high is None else high

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    cmap = cmap or cm.gray
    plt.imsave(filename, data, cmap=cmap, vmin=low, vmax=high, dpi=dpi)
    plt.clf()

def convert_npy_to_8bit_image(filename, output=None, cmap=None, cmin=None, cmax=None, low=None, high=None):
    low = 0 if low is None else low
    high = 255 if high is None else high

    output = output or '{}.png'.format(os.path.splitext(filename)[0])

    data = numpy.load(filename)
    data = data[: , : , 1]

    bytedata = convert_8bit(data, cmin, cmax, low, high)
    save_image(output, bytedata, cmap, low, high)

def save_image_with_spots(filename, data, spots, cmap=None, low=None, high=None, dpi=100):
    """Generate an image with spots.

    Args:
        filename (str): An output file name.
        data (ndarray): An image data.
        spots (list) A list of spots. spots are represented as a tuple consisting of
            height, center_x, center_y, width_x, width_y, and background.
        cmap (optinal): A color map.
        low, high (float, optional): vmin and vmax define the range of the colormap.
            By default, the colormap covers the complete value range of the data.
        dpi (float, optional): Defaults to 100.

    """
    data = numpy.asarray(data)
    low = data.min() if low is None else low
    high = data.max() if high is None else high

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.patches as patches
    cmap = cmap or cm.gray

    fig = plt.figure()
    m, n = data.shape
    fig.set_size_inches((m / dpi, n / dpi))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data, interpolation='none', cmap=cmap, vmin=low, vmax=high)

    for spot in spots:
        (_, center_x, center_y, width_x, width_y, _) = spot
        # radius = max(width_x, width_y)
        # c = plt.Circle((center_y, center_x), radius, color='red', linewidth=1, fill=False)
        c = patches.Ellipse((center_y, center_x), width_y, width_x, color='red', linewidth=1, fill=False)
        ax.add_patch(c)

    plt.savefig(filename)
    plt.clf()

def show_with_spots(data, spots=None, blobs=None, cmap=None, low=None, high=None, dpi=100):
    data = numpy.asarray(data)
    low = data.min() if low is None else low
    high = data.max() if high is None else high

    # import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.patches as patches
    cmap = cmap or cm.gray

    fig = plt.figure()
    m, n = data.shape
    fig.set_size_inches((m / dpi, n / dpi))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data, interpolation='none', cmap=cmap, vmin=low, vmax=high)

    if blobs is not None:
        for blob in blobs:
            (x, y, r) = blob

            # c = plt.Circle((y, x), r, color='green', linewidth=1, fill=False)
            # ax.add_patch(c)

            x0, x1 = int(x - 2 * r), int(x + 2 * r)
            y0, y1 = int(y - 2 * r), int(y + 2 * r)
            x0, x1 = max(0, x0), min(data.shape[0], x1)
            y0, y1 = max(0, y0), min(data.shape[1], y1)
            c = patches.Rectangle((y0, x0), y1 - y0, x1 - x0, color='green', linewidth=1, fill=False)
            ax.add_patch(c)

    if spots is not None:
        snratio = spots.T[0] / spots.T[-1]
        imin, imax = snratio.min(), snratio.max()
        for spot in spots:
            (height, center_x, center_y, width_x, width_y, bg) = spot
            intensity = (height / bg - imin) / (imax - imin)
            c = patches.Ellipse((center_y, center_x), width_y * 2, width_x * 2, color=cm.plasma(intensity), linewidth=1, fill=False)
            ax.add_patch(c)

    plt.show()
    plt.clf()
