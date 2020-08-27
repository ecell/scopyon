import dataclasses

import numpy
import scipy.optimize

from logging import getLogger, DEBUG
_log = getLogger(__name__)

__all__ = ["blob_detection", "spot_detection"]


def blob_detection(data, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5):
    """Finds blobs in the given image.
    See also http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log

    Note:
        Requires `scikit-image`.

    Args:
        data (ndarray): An image data.
        min_sigma (float, optional): The minimum standard deviation.
            Keep this low to detect smaller blobs. Defaults to 1.
        max_sigma (float, optional): The maximum standard deviation.
            Keep this high to detect larger blobs. Defaults to 50.
        num_sigma (int, optional): The number of intermediate values between `min_sigma` and `max_sigma`.
            Defaults to 10.
        threshold (float, optional): The absolute lower bound for scale space maxima.
            Reduce this to detect blobs with less intensities.
        overlap (float, optional): A value between 0 and 1.

    Returns:
        ndarray: Blobs detected.
            Each row represents coordinates and the standard deviation, `(x, y, r)`.
    """
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
    bg = numpy.tile(numpy.arange(m)[:, numpy.newaxis], (1, n)) * a5 + numpy.tile(numpy.arange(n), (m, 1)) * a6 + a7
    return roi.astype(dtype=numpy.float64) - bg, bg.sum()

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

@dataclasses.dataclass
class LoggingMessage:
    message: str = ""
    level: int = DEBUG

def __spot_detection(data, roi_size, blob):
    x, y, _ = blob

    x0, x1 = int(x - roi_size), int(x + roi_size) + 1
    y0, y1 = int(y - roi_size), int(y + roi_size) + 1
    x0, x1 = max(0, x0), min(data.shape[0], x1)
    y0, y1 = max(0, y0), min(data.shape[1], y1)
    roi = data[x0: x1, y0: y1]

    if roi.sum() <= 0:
        return LoggingMessage("spot_detection skip a blob due to the low signal.")

    roi_, bg = background(roi)
    if roi_ is None:
        return LoggingMessage("spot_detection skip a blob due to the failure in background.")
    res = fitgaussian(roi_, roi_size)
    if res is None:
        return LoggingMessage("spot_detection skip a blob due to the failure in fitgaussian.")

    (height, center_x, center_y, sigma) = res
    if not (0 <= center_x < roi.shape[0]
            and 0 <= center_y < roi.shape[1]):
        return LoggingMessage("spot_detection skip a blob due to invalid parameters fitted.")

    intensity = gaussian(*res)(*numpy.indices(roi.shape)).sum()
    center_x += x0
    center_y += y0
    return (center_x, center_y, intensity, bg, height, sigma)

def spot_detection(data, roi_size=6, blobs=None, processes=None, **kwargs):
    """Finds spots in the given image.

    Note:
        Requires `scipy`.

    Args:
        data (ndarray): An image data.
        roi_size (float, optional): A default value of a half of the ROI size.
            Defaults to 6.
        blobs (ndarray, optional): Blobs. Defaults to `None`. See also `blob_detection`.

    Returns:
        ndarray: Spots detected.
            Each row represents center position x, y, intensity, background, height, and sigma,
            `(center_x, center_y, intensity, bg, height, sigma)`.
    """

    if blobs is None:
        blobs = blob_detection(data, **kwargs)

    if processes is not None and processes > 1:
        with Pool(processes) as pool:
            spots = pool.map(functools.partial(__spot_detection, data=data, roi_size=roi_size), blobs)
        spots = [spot for spot in spots if not isinstance(spot, LoggingMessage)]
    else:
        spots = []
        for blob in blobs:
            spot = __spot_detection(data, roi_size, blob)
            if isinstance(spot, LoggingMessage):
                _log.log(spot.level, spot.message)
            else:
                spots.append(spot)

    _log.info('{} spot(s) were detected'.format(len(spots)))
    spots = numpy.array(spots)
    return spots
