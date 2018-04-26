import numpy

import scipy.optimize

from logging import getLogger
_log = getLogger(__name__)


def gaussian(height, center_x, center_y, width_x, width_y, bg):
    """Returns a gaussian function with the given parameters
    http://scipy-cookbook.readthedocs.io/items/FittingData.html

    """
    width_x = float(width_x)
    width_y = float(width_y)
    bg = float(bg)
    return lambda x, y: height / (2 * numpy.pi * numpy.sqrt(width_x ** 2 + width_y ** 2)) * numpy.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2) + bg
    # return lambda x, y: height * numpy.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2) + bg

def moments(data):
    """Returns (height, x, y, width_x, width_y, bg)
    the gaussian parameters of a 2D distribution by calculating its
    moments
    http://scipy-cookbook.readthedocs.io/items/FittingData.html

    """
    total = data.sum()
    X, Y = numpy.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[: , int(y)]
    width_x = numpy.sqrt(numpy.abs((numpy.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = numpy.sqrt(numpy.abs((numpy.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    bg = data.min()
    height = data.max() - bg
    height *= (2 * numpy.pi * numpy.sqrt(width_x ** 2 + width_y ** 2))
    return height, x, y, width_x, width_y, bg

def gaussian_pixelized(shape, *p, ndiv=5):
    """
    Returns:
        ndarray: Return an array with shape (m, n)

    """
    (m, n) = shape
    X, Y = numpy.meshgrid(numpy.linspace(0, m, m * ndiv + 1), numpy.linspace(0, n, n * ndiv + 1))
    X, Y = X.T, Y.T
    raw = gaussian(*p)(X, Y)
    ret = numpy.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ret[i, j] = numpy.average(raw[i * ndiv: (i + 1) * ndiv + 1, j * ndiv : (j + 1) * ndiv + 1])
    return ret

def fitgaussian(data, opt=1):
    """Returns (height, x, y, width_x, width_y, bg)
    the gaussian parameters of a 2D distribution found by a fit
    http://scipy-cookbook.readthedocs.io/items/FittingData.html

    """
    params = moments(data)

    if opt == 0:
        return params

    elif opt == 1:
        # error_func = lambda p: numpy.ravel(gaussian(*p)(*numpy.indices(data.shape)) - data)
        error_func = lambda p: numpy.ravel(gaussian_pixelized(data.shape, *p, ndiv=2) - data)
        p, suc = scipy.optimize.leastsq(error_func, params)
        if suc not in (1, 2, 3, 4):
            return None
        return p

    elif opt == 2:
        bounds = ((0, 0, 0, 0, 0, 0), (numpy.inf, data.shape[0], data.shape[1], data.shape[0], data.shape[1], numpy.inf))
        error_func = lambda p: numpy.ravel(gaussian(*p)(*numpy.indices(data.shape)) - data)
        # error_func = lambda p: numpy.ravel(gaussian_pixelized(data.shape, *p, ndiv=5) - data)
        res = scipy.optimize.least_squares(error_func, params, bounds=bounds)
        if res.success <= 0:
            return None
        return res.x

    else:
        error_func = lambda p: numpy.ravel(gaussian(p[0], p[1], p[2], params[3], params[4], p[1])(*numpy.indices(data.shape)) - data)
        p, suc = scipy.optimize.leastsq(error_func, (params[0], params[1], params[2], params[5]))
        if suc not in (1, 2, 3, 4):
            return None
        return (p[0], p[1], p[2], params[3], params[4], p[3])

def blob_detection(data, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5):
    try:
        from skimage.feature import blob_log
    except ImportError:
        raise ImportError("No module named 'skimage'. 'spot_detection' requires 'scikit-image'")

    ## Laplacian of Gaussian
    blobs = blob_log(
        data, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    blobs[: , 2] = blobs[: , 2] * numpy.sqrt(2)
    _log.info('{} blob(s) were detected'.format(len(blobs)))
    return blobs

def spot_detection(data, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5, blobs=None, opt=1):
    if blobs is None:
        blobs = blob_detection(
            data, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap)

    spots = []
    for blob in blobs:
        x, y, r = blob

        # x0, x1 = int(x - 1.5 * r), int(x + 1.5 * r) + 1
        # y0, y1 = int(y - 1.5 * r), int(y + 1.5 * r) + 1
        x0, x1 = int(x - 2 * r), int(x + 2 * r) + 1
        y0, y1 = int(y - 2 * r), int(y + 2 * r) + 1
        x0, x1 = max(0, x0), min(data.shape[0], x1)
        y0, y1 = max(0, y0), min(data.shape[1], y1)

        roi = data[x0: x1, y0: y1]
        if roi.sum() <= 0:
            continue

        ## Gaussian-fit to i-th spot
        ret = fitgaussian(roi, opt)
        if ret is None:
            continue

        (height, center_x, center_y, width_x, width_y, bg) = ret
        width_x = abs(width_x)
        width_y = abs(width_y)

        if not (
            height > 0
            # and 0 <= center_x < data.shape[0]
            # and 0 <= center_y < data.shape[1]
            and 0 <= center_x < roi.shape[0]
            and 0 <= center_y < roi.shape[1]
            and bg > 0):
            continue

        center_x += x0
        center_y += y0
        spots.append((height, center_x, center_y, width_x, width_y, bg))

    _log.info('{} spot(s) were detected'.format(len(spots)))
    return numpy.array(spots)


if __name__ == "__main__":
    import sys
    import matplotlib.pylab as plt


    for filename in sys.argv[1: ]:
        data = numpy.load(filename)
        data = data[: , : , 1]

        # spot-detection
        spots = spot_detection(data, min_sigma=2.0, max_sigma=4.0, num_sigma=20, threshold=10, overlap=0.5)

        # show figures
        dpi = 100
        fig = plt.figure()
        m, n = data.shape
        fig.set_size_inches((m / dpi, n / dpi))
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(data, interpolation='none', cmap='gray')

        for spot in spots:
            (height, center_x, center_y, width_x, width_y, bg) = spot

            radius = max(width_x, width_y)
            c = plt.Circle((center_x, center_y), radius, color='red', linewidth=1, fill=False)
            ax.add_patch(c)

        plt.savefig('out.png')
