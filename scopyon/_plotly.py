import numpy

#XXX: Never import plotly here. Do inside each function.

__all__ = ['show']


def __get_shape(shape):
    x, y, sigma, c = shape['x'], shape['y'], shape['sigma'], shape['color']
    return dict(type='rect', x0=x - sigma, y0=y - sigma, x1=x + sigma, y1=y + sigma, line=dict(color=c))

def show(img, shapes=None):
    """Show an image.

    Note:
        Requires `plotly.express`.

    Args:
        img (ndarray): An image data to be shown.
        shapes (list, optional): A list of shapes.
            shape is a dictionary consisting of `x`, `y`, `sigma` and `color`.
            `sigma` is a half size of the box (square).
    """
    if not isinstance(img, numpy.ndarray):
        raise TypeError("'img' must be an Image or array.")

    if shapes is not None:
        shapes = [__get_shape(shape) for shape in shapes]

    import plotly.express as px
    if img.ndim == 2:
        fig = px.imshow(img, color_continuous_scale='gray')
    elif img.ndim == 3:
        fig = px.imshow(img)
    else:
        raise ValueError("'img' has wrong dimension.")
    if shapes is not None:
        fig.update_layout(shapes=shapes)
    fig.update_layout(xaxis=dict(range=[0, img.shape[0]]), yaxis=dict(range=[0, img.shape[1]]))
    fig.show()
