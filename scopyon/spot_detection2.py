import numpy
import scipy.optimize

from logging import getLogger
_log = getLogger(__name__)


def mean_background(roi):
    m, n = roi.shape
    left = roi[0, : -1].sum()
    right = roi[-1, : -1].sum()
    bottom = roi[: -1, 0].sum()
    top = roi[: -1, -1].sum()
    tot = left + right + bottom + top - (roi[0, 0] + roi[0, -1] + roi[-1, 0] + roi[-1, -1])
    a7 = tot / (2 * (m + n - 2))
    # return roi.astype(dtype=numpy.float64) - a7
    return a7

def planar_background(roi):
    m, n = roi.shape
    a7_init = mean_background(roi)
    a5_init = a6_init = 0.0
    def error_func(p):
        a5, a6, a7 = p
        left = roi[0, :] - (numpy.arange(n) * a6 + a7)
        right = roi[-1, :] - ((m - 1) * a5 + numpy.arange(n) * a6 + a7)
        bottom = roi[:, 0] - (numpy.arange(m) * a5 + a7)
        top = roi[:, -1] - (numpy.arange(m) * a5 + (n - 1) * a6 + a7)
        return numpy.concatenate([left, right, bottom, top])
        # return (left ** 2).sum() + (right ** 2).sum() + (bottom ** 2).sum() + (top ** 2).sum()
    res = scipy.optimize.least_squares(error_func, (a5_init, a6_init, a7_init))
    if res.success <= 0:
        return None
    return res.x

def background(roi):
    m, n = roi.shape
    a5, a6, a7 = planar_background(roi)
    return roi.astype(dtype=numpy.float64) - (numpy.tile(numpy.arange(m)[:, numpy.newaxis], (1, n)) * a5 + numpy.tile(numpy.arange(n), (m, 1)) * a6 + a7)


def gaussian(a1, a2, a3, a4):
    return lambda x, y: (
        a1 * numpy.exp(-((x - a2) ** 2 + (y - a3) ** 2) / a4)) 

def weighted_com(data):
    total = data.sum()
    X, Y = numpy.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    return (x, y)

def fitgaussian(data, roi_size):
    a2_init, a3_init = weighted_com(data)
    a1_init = 255
    a4_init = roi_size / 2
    error_func = lambda p: numpy.ravel(gaussian(*p)(*numpy.indices(data.shape)) - data)
    res = scipy.optimize.least_squares(error_func, (a1_init, a2_init, a3_init, a4_init))
    if res.success <= 0:
        return None
    return res.x

def spot_detection(data, blobs, roi_size=6):
    """Finds spots in the given image.

    Args:
        data (ndarray): An image data.
        blobs (ndarray, optional): Blobs. Defaults to `None`. See also `blob_detection`.
        roi_size (float, optional): A default value of a half of the ROI size.
            Defaults to 6.

    Returns:
        ndarray: Spots detected.
            Each row represents height, center position x, y, width x, y, and background,
            `(intensity, height, center_x, center_y, sigma)`.
    """

    spots = []
    for blob in blobs:
        x, y, _ = blob

        x0, x1 = int(x - roi_size), int(x + roi_size) + 1
        y0, y1 = int(y - roi_size), int(y + roi_size) + 1
        x0, x1 = max(0, x0), min(data.shape[0], x1)
        y0, y1 = max(0, y0), min(data.shape[1], y1)
        roi = data[x0: x1, y0: y1]

        if roi.sum() <= 0:
            _log.debug("spot_detection skip a blob due to the low signal.")
            continue

        roi_ = background(roi)
        if roi_ is None:
            _log.debug("spot_detection skip a blob due to the failure in background.")
            continue
        res = fitgaussian(roi_, roi_size)
        if res is None:
            _log.debug("spot_detection skip a blob due to the failure in fitgaussian.")
            continue

        (height, center_x, center_y, sigma) = res
        if not (
            0 <= center_x < roi.shape[0]
            and 0 <= center_y < roi.shape[1]):
            _log.debug("spot_detection skip a blob due to invalid parameters fitted.")
            continue

        intensity = gaussian(*res)(*numpy.indices(roi.shape)).sum()
        center_x += x0
        center_y += y0
        spots.append((intensity, height, center_x, center_y, sigma))

    _log.info('{} spot(s) were detected'.format(len(spots)))
    spots = numpy.array(spots)
    return spots
