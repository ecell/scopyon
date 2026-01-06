import numpy

__all__ = ['show']


def __get_shape(shape):
    import matplotlib.patches as patches
    row, column, sigma, c = shape['x'], shape['y'], shape['sigma'], shape['color']
    x, y = column, row  # imshow
    return patches.Rectangle((x - sigma, y - sigma), 2 * sigma, 2 * sigma, color=c, linewidth=1, fill=False)

def show(img, shapes=None):
    """Show an image.

    Args:
        img (ndarray): An image data to be shown.
        shapes (list, optional): A list of shapes.
            shape is a dictionary consisting of `x` (row), `y` (column), `sigma` and `color`.
            `sigma` is a half size of the box (square).
    """
    if not isinstance(img, numpy.ndarray):
        raise TypeError("'img' must be an Image or array.")

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    if img.ndim == 2:
        plt.imshow(img, interpolation='none', cmap='gray')
    elif img.ndim == 3:
        plt.imshow(img, interpolation='none')
    else:
        raise ValueError("'img' has wrong dimension.")
    if shapes is not None:
        for shape in shapes:
            ax.add_patch(__get_shape(shape))
    plt.show()
