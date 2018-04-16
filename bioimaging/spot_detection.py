import itertools
import math

import numpy

import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_laplace


def get_high_intensity_peaks(image, mask, num_peaks):
    """ Return the highest intensity peak coordinates."""
     # get coordinates of peaks
    coord = numpy.nonzero(mask)

    # select num_peaks peaks
    if len(coord[0]) > num_peaks:
        intensities = image[coord]
        idx_maxsort = np.argsort(intensities)
        coord = numpy.transpose(coord)[idx_maxsort][-num_peaks: ]
    else:
        coord = numpy.column_stack(coord)

    # Higest peak first
    return coord[: : -1]

def peak_local_max(
        image, min_distance=1, threshold_abs=None, threshold_rel=None, exclude_border=True,
        indices=True, num_peaks=numpy.inf, footprint=None, labels=None,
        num_peaks_per_label=numpy.inf):
    if isinstance(exclude_border, bool):
        exclude_border = min_distance if exclude_border else 0

    out = numpy.zeros_like(image, dtype=numpy.bool)

    # In the case of labels, recursively build and return an output
    # operating on each label separately
    if labels is not None:
        label_values = numpy.unique(labels)

        # Reorder label values to have consecutive integers (no gaps)
        if numpy.any(numpy.diff(label_values) != 1):
            mask = labels >= 1
            labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
        labels = labels.astype(numpy.int32)

        # New values for new ordering
        label_values = numpy.unique(labels)
        for label in label_values[label_values != 0]:
            maskim = (labels == label)
            out += peak_local_max(
                image * maskim, min_distance=min_distance,
                threshold_abs=threshold_abs,
                threshold_rel=threshold_rel,
                exclude_border=exclude_border,
                indices=False, num_peaks=num_peaks_per_label,
                footprint=footprint, labels=None)

        # Select highest intensities (num_peaks)
        coordinates = get_high_intensity_peaks(image, out, num_peaks)

        if indices is True:
            return coordinates
        else:
            nd_indices = tuple(coordinates.T)
            out[nd_indices] = True
            return out

    if numpy.all(image == image.flat[0]):
        if indices is True:
            return numpy.empty((0, 2), numpy.int)
        else:
            return out

    # Non-maximum filter
    if footprint is not None:
        image_max = ndimage.maximum_filter(image, footprint=footprint, mode='constant')
    else:
        size = 2 * min_distance + 1
        image_max = ndimage.maximum_filter(image, size=size, mode='constant')

    mask = (image == image_max)

    if exclude_border:
        # zero out the image borders
        for i in range(mask.ndim):
            mask = mask.swapaxes(0, i)
            remove = (footprint.shape[i] if footprint is not None else 2 * exclude_border)
            mask[: remove // 2] = mask[-remove // 2: ] = False
            mask = mask.swapaxes(0, i)

    # find top peak candidates above a threshold
    thresholds = []
    if threshold_abs is None:
        threshold_abs = image.min()
    thresholds.append(threshold_abs)
    if threshold_rel is not None:
        thresholds.append(threshold_rel * image.max())
    if thresholds:
        mask &= image > max(thresholds)

    # Select highest intensities (num_peaks)
    coordinates = get_high_intensity_peaks(image, mask, num_peaks)

    if indices is True:
        return coordinates
    else:
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True
        return out

def blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs. Returns a float
    representing fraction of overlapped area.

    Args:
        blob1 (sequence): A sequence of ``(y,x,sigma)``, where ``x,y`` are
            coordinates of blob and sigma is the standard deviation of the Gaussian
            kernel which detected the blob.
        blob2 (sequence): A sequence of ``(y,x,sigma)``, where ``x,y`` are
            coordinates of blob and sigma is the standard deviation of the Gaussian
            kernel which detected the blob.

    Returns:
        float: Fraction of overlapped area.

    """
    root2 = math.sqrt(2)

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[2] * root2
    r2 = blob2[2] * root2

    d = math.hypot(blob1[0] - blob2[0], blob1[1] - blob2[1])

    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = numpy.clip(ratio1, -1, 1)
    acos1 = numpy.arccos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = numpy.clip(ratio2, -1, 1)
    acos2 = numpy.arccos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * math.sqrt(abs(a * b * c * d))

    return area / (math.pi * (min(r1, r2) ** 2))

def prune_blobs(blobs_array, overlap):
    """Eliminated blobs with area overlap.

    Args:
        blobs_array (ndarray): A 2d array with each row representing 3 values,
            ``(y,x,sigma)`` where ``(y,x)`` are coordinates of the blob and
            ``sigma`` is the standard deviation of the Gaussian kernel which
            detected the blob.
        overlap (float): A value between 0 and 1. If the fraction of area
            overlapping for 2 blobs is greater than `overlap` the smaller blob
            is eliminated.

    Returns:
        ndarray: `array` with overlapping blobs removed.

    """
    # iterating again might eliminate more blobs, but one iteration suffices
    # for most cases
    for blob1, blob2 in itertools.combinations(blobs_array, 2):
        if blob_overlap(blob1, blob2) > overlap:
            if blob1[2] > blob2[2]:
                blob2[2] = -1
            else:
                blob1[2] = -1

    # return blobs_array[blobs_array[: , 2] > 0]
    return numpy.array([b for b in blobs_array if b[2] > 0])

def spot_detection(data, sigma_list, threshold, overlap):
    # # detecting spots
    #     if log_scale:
    #         start, stop = math.log(min_sigma, 10), math.log(max_sigma, 10)
    #         sigma_list = numpy.logspace(start, stop, num_sigma)
    #     else:
    #         sigma_list = numpy.linspace(min_sigma, max_sigma, num_sigma)

    # computing gaussian laplace
    # s**2 provides scale invariance
    gl_images = [-ndimage.gaussian_laplace(data, s) * (s * s) for s in sigma_list]
    image_cube = numpy.dstack(gl_images)

    local_maxima = peak_local_max(
        image_cube, threshold_abs=threshold, footprint=numpy.ones((3, 3, 3)),
        threshold_rel=0.0, exclude_border=False)

    # Catch no peaks
    if local_maxima.size == 0:
        return numpy.empty((0, 3))

    # Convert local_maxima to float64
    lm = local_maxima.astype(numpy.float64)

    # Convert the last index to its corresponding scale value
    lm[: , 2] = sigma_list[local_maxima[: , 2]]
    local_maxima = lm

    spots = prune_blobs(local_maxima, overlap)
    return spots

# def convert_image2numpy(ligand, cell_index, file_in, file_out) :
#     # set index
#     index = 0
#     spot_index = 0
# 
#     # set spot-data
#     spot_data = []
# 
#     while (True) :
#         # get arrays
#         try :
#             image_array = numpy.load(file_in + '/image_%07d.npy' % (index))
#             true_array = numpy.load(file_in + '/true_%07d.npy' % (index))
#         except Exception :
#             print('error : file not found at %02d-th index' % (index))
#             break
# 
#         # image-array
#         obs_image = image_array[:,:,1]
#         exp_image = image_array[:,:,0]
# 
#         # spot-detection
#         sigma = numpy.linspace(4.0, 6.0, 20)
#         spots = spot_detection(obs_image, sigma_list=sigma, threshold=10, overlap=0.5)
# 
#         for i in range(len(spots)) :
#             x, y, r = spots[i]
# 
#             x0, x1 = int(x-2*r), int(x+2*r)
#             y0, y1 = int(y-2*r), int(y+2*r)
# 
#             image = obs_image[x0:x1,y0:y1]
#             true_data = spot_candidates(true_array, x0,x1,y0,y1)
# 
#             if (image.sum() > 0) :
# 
#                 # Gaussian-fit to i-th spot
#                 reco_spot = fitgaussian(image)
#                 reco_N0 = reco_spot[0] # ADC-counts
#                 reco_x0 = reco_spot[1] + x0 # pixel
#                 reco_y0 = reco_spot[2] + y0 # pixel
#                 reco_wx = reco_spot[3]
#                 reco_wy = reco_spot[4]
#                 reco_bg = reco_spot[5]
# 
#             # get true-property
#             #true_spot = numpy.full((7), -100, dtype=int)
#             true_spot = get_true_spot(reco_spot, true_data)
#             time, mol_id, mol_st, pho_st = true_spot[0:4]
#             true_x0, true_y0, true_z0 = true_spot[4:7]
# 
#             if (true_x0 < 512 and true_y0 < 512) :
#                 true_N0 = exp_image[int(true_x0),int(true_y0)]
#             else :
#                 true_N0 = -1
# 
#             # initial-cut
#             if (time>-1 and \
#                 reco_x0>0 and reco_x0<512 and \
#                 reco_y0>0 and reco_y0<512 and \
#                 reco_wx>0 and reco_wy>0 and \
#                 reco_N0>0 and reco_bg>0) :
# 
#                 variables = []
# 
#             # get index
#             variables.append(ligand)
#             variables.append(cell_index)
#             variables.append(spot_index)
# 
#             # get true-variables
#             variables.append(time) # sec
#             variables.append(mol_id) # Spatiocyte-molecule ID
#             variables.append(mol_st) # Molecule state
#             variables.append(pho_st) # Photon-emission state
#             variables.append(true_N0)
#             variables.append(true_x0) # pixel
#             variables.append(true_y0) # pixel
#             variables.append(true_z0) # nm
# 
#             # get reconstructed-variables
#             variables.append(reco_N0) # ADC-counts
#             variables.append(reco_x0) # pixel
#             variables.append(reco_y0) # pixel
#             variables.append(reco_wx) # pixel
#             variables.append(reco_wy) # pixel
#             variables.append(reco_bg) # ADC-counts
# 
#             spot_data.append(variables)
# 
#             spot_index += 1
# 
#         index += 1
# 
#     # save the spot-data as numpy-file
#     array = numpy.array(spot_data)
#     numpy.save(file_out, array)



if __name__ == "__main__":
    import sys
    import matplotlib.pylab as plt


    for filename in sys.argv[1: ]:
        data = numpy.load(filename)
        data = data[: , : , 1]

        # spot-detection
        sigma = numpy.linspace(2.0, 4.0, 20)
        spots = spot_detection(data, sigma_list=sigma, threshold=10, overlap=0.5)
        print('{} spot(s) were detected [{}]'.format(len(spots), filename))

        # show figures
        fig, ax = plt.subplots()

        ax.imshow(data, interpolation='none', cmap='gray')

        for spot in spots:
            y, x, r = spot
            c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
            ax.add_patch(c)

        ax.set_axis_off()

        plt.tight_layout()
        plt.show()
