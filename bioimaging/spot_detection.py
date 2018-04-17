import itertools
import math

import numpy

import scipy.optimize
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_laplace

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

def fitgaussian(data, opt=1):
    """Returns (height, x, y, width_x, width_y, bg)
    the gaussian parameters of a 2D distribution found by a fit
    http://scipy-cookbook.readthedocs.io/items/FittingData.html
    
    """
    params = moments(data)

    def error_func(p):
        return numpy.ravel(gaussian(*p)(*numpy.indices(data.shape)) - data)

        m, n = data.shape
        ndiv = 5
        X, Y = numpy.meshgrid(numpy.linspace(0, m, ndiv + 1), numpy.arange(0, n, ndiv + 1))
        # X, Y = numpy.meshgrid(numpy.arange(0, m + m / ndiv, m / ndiv), numpy.arange(0, n + n / ndiv, n / ndiv))
        X, Y = X.T, Y.T
        res = gaussian(*p)(X, Y)
        data_ = numpy.zeros((m, n))
        for i in range(m):
            for j in range(n):
                data_[i, j] = numpy.average(res[i * ndiv: (i + 1) * ndiv + 1, j * ndiv : (j + 1) * ndiv + 1])
        return numpy.ravel(data_ - data)

    if opt == 0:
        return params

    elif opt == 1:
        p, suc = scipy.optimize.leastsq(error_func, params)
        if suc not in (1, 2, 3, 4):
            return None
        return p

    elif opt == 2:
        bounds = ((0, 0, 0, 0, 0, 0), (numpy.inf, data.shape[0], data.shape[1], data.shape[0], data.shape[1], numpy.inf))
        res = scipy.optimize.least_squares(error_func, params, bounds=bounds)
        if res.success <= 0:
            return None
        return res.x

    else:
        errorfunction = lambda p: numpy.ravel(gaussian(p[0], p[1], p[2], params[3], params[4], p[1])(*numpy.indices(data.shape)) - data)
        p, suc = scipy.optimize.leastsq(errorfunction,(params[0], params[1], params[2], params[5]))
        if suc not in (1, 2, 3, 4):
            return None
        return (p[0], p[1], p[2], params[3], params[4], p[1])

# def get_high_intensity_peaks(image, mask, num_peaks):
#     """ Return the highest intensity peak coordinates."""
#      # get coordinates of peaks
#     coord = numpy.nonzero(mask)
# 
#     # select num_peaks peaks
#     if len(coord[0]) > num_peaks:
#         intensities = image[coord]
#         idx_maxsort = np.argsort(intensities)
#         coord = numpy.transpose(coord)[idx_maxsort][-num_peaks: ]
#     else:
#         coord = numpy.column_stack(coord)
# 
#     # Higest peak first
#     return coord[: : -1]
# 
# def peak_local_max(
#         image, min_distance=1, threshold_abs=None, threshold_rel=None, exclude_border=True,
#         indices=True, num_peaks=numpy.inf, footprint=None, labels=None,
#         num_peaks_per_label=numpy.inf):
#     if isinstance(exclude_border, bool):
#         exclude_border = min_distance if exclude_border else 0
# 
#     out = numpy.zeros_like(image, dtype=numpy.bool)
# 
#     # In the case of labels, recursively build and return an output
#     # operating on each label separately
#     if labels is not None:
#         label_values = numpy.unique(labels)
# 
#         # Reorder label values to have consecutive integers (no gaps)
#         if numpy.any(numpy.diff(label_values) != 1):
#             mask = labels >= 1
#             labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
#         labels = labels.astype(numpy.int32)
# 
#         # New values for new ordering
#         label_values = numpy.unique(labels)
#         for label in label_values[label_values != 0]:
#             maskim = (labels == label)
#             out += peak_local_max(
#                 image * maskim, min_distance=min_distance,
#                 threshold_abs=threshold_abs,
#                 threshold_rel=threshold_rel,
#                 exclude_border=exclude_border,
#                 indices=False, num_peaks=num_peaks_per_label,
#                 footprint=footprint, labels=None)
# 
#         # Select highest intensities (num_peaks)
#         coordinates = get_high_intensity_peaks(image, out, num_peaks)
# 
#         if indices is True:
#             return coordinates
#         else:
#             nd_indices = tuple(coordinates.T)
#             out[nd_indices] = True
#             return out
# 
#     if numpy.all(image == image.flat[0]):
#         if indices is True:
#             return numpy.empty((0, 2), numpy.int)
#         else:
#             return out
# 
#     # Non-maximum filter
#     if footprint is not None:
#         image_max = ndimage.maximum_filter(image, footprint=footprint, mode='constant')
#     else:
#         size = 2 * min_distance + 1
#         image_max = ndimage.maximum_filter(image, size=size, mode='constant')
# 
#     mask = (image == image_max)
# 
#     if exclude_border:
#         # zero out the image borders
#         for i in range(mask.ndim):
#             mask = mask.swapaxes(0, i)
#             remove = (footprint.shape[i] if footprint is not None else 2 * exclude_border)
#             mask[: remove // 2] = mask[-remove // 2: ] = False
#             mask = mask.swapaxes(0, i)
# 
#     # find top peak candidates above a threshold
#     thresholds = []
#     if threshold_abs is None:
#         threshold_abs = image.min()
#     thresholds.append(threshold_abs)
#     if threshold_rel is not None:
#         thresholds.append(threshold_rel * image.max())
#     if thresholds:
#         mask &= image > max(thresholds)
# 
#     # Select highest intensities (num_peaks)
#     coordinates = get_high_intensity_peaks(image, mask, num_peaks)
# 
#     if indices is True:
#         return coordinates
#     else:
#         nd_indices = tuple(coordinates.T)
#         out[nd_indices] = True
#         return out
# 
# def blob_overlap(blob1, blob2):
#     """Finds the overlapping area fraction between two blobs. Returns a float
#     representing fraction of overlapped area.
# 
#     Args:
#         blob1 (sequence): A sequence of ``(y,x,sigma)``, where ``x,y`` are
#             coordinates of blob and sigma is the standard deviation of the Gaussian
#             kernel which detected the blob.
#         blob2 (sequence): A sequence of ``(y,x,sigma)``, where ``x,y`` are
#             coordinates of blob and sigma is the standard deviation of the Gaussian
#             kernel which detected the blob.
# 
#     Returns:
#         float: Fraction of overlapped area.
# 
#     """
#     root2 = math.sqrt(2)
# 
#     # extent of the blob is given by sqrt(2)*scale
#     r1 = blob1[2] * root2
#     r2 = blob2[2] * root2
# 
#     d = math.hypot(blob1[0] - blob2[0], blob1[1] - blob2[1])
# 
#     if d > r1 + r2:
#         return 0
# 
#     # one blob is inside the other, the smaller blob must die
#     if d <= abs(r1 - r2):
#         return 1
# 
#     ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
#     ratio1 = numpy.clip(ratio1, -1, 1)
#     acos1 = numpy.arccos(ratio1)
# 
#     ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
#     ratio2 = numpy.clip(ratio2, -1, 1)
#     acos2 = numpy.arccos(ratio2)
# 
#     a = -d + r2 + r1
#     b = d - r2 + r1
#     c = d + r2 - r1
#     d = d + r2 + r1
#     area = r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * math.sqrt(abs(a * b * c * d))
# 
#     return area / (math.pi * (min(r1, r2) ** 2))
# 
# def prune_blobs(blobs_array, overlap):
#     """Eliminated blobs with area overlap.
# 
#     Args:
#         blobs_array (ndarray): A 2d array with each row representing 3 values,
#             ``(y,x,sigma)`` where ``(y,x)`` are coordinates of the blob and
#             ``sigma`` is the standard deviation of the Gaussian kernel which
#             detected the blob.
#         overlap (float): A value between 0 and 1. If the fraction of area
#             overlapping for 2 blobs is greater than `overlap` the smaller blob
#             is eliminated.
# 
#     Returns:
#         ndarray: `array` with overlapping blobs removed.
# 
#     """
#     # iterating again might eliminate more blobs, but one iteration suffices
#     # for most cases
#     for blob1, blob2 in itertools.combinations(blobs_array, 2):
#         if blob_overlap(blob1, blob2) > overlap:
#             if blob1[2] > blob2[2]:
#                 blob2[2] = -1
#             else:
#                 blob1[2] = -1
# 
#     # return blobs_array[blobs_array[: , 2] > 0]
#     return numpy.array([b for b in blobs_array if b[2] > 0])

# def blob_detection(data, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5):
#     # # detecting spots
#     #     if log_scale:
#     #         start, stop = math.log(min_sigma, 10), math.log(max_sigma, 10)
#     #         sigma_list = numpy.logspace(start, stop, num_sigma)
#     #     else:
#     #         sigma_list = numpy.linspace(min_sigma, max_sigma, num_sigma)
# 
#     # computing gaussian laplace
#     # s**2 provides scale invariance
#     gl_images = [-ndimage.gaussian_laplace(data, s) * (s * s) for s in sigma_list]
#     image_cube = numpy.dstack(gl_images)
# 
#     local_maxima = peak_local_max(
#         image_cube, threshold_abs=threshold, footprint=numpy.ones((3, 3, 3)),
#         threshold_rel=0.0, exclude_border=False)
# 
#     # Catch no peaks
#     if local_maxima.size == 0:
#         return numpy.empty((0, 3))
# 
#     # Convert local_maxima to float64
#     lm = local_maxima.astype(numpy.float64)
# 
#     # Convert the last index to its corresponding scale value
#     lm[: , 2] = sigma_list[local_maxima[: , 2]]
#     local_maxima = lm
# 
#     spots = prune_blobs(local_maxima, overlap)
#     return spots

def blob_detection(data, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5):
    from skimage.feature import blob_log

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
