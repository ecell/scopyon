import numpy

from logging import getLogger
_log = getLogger(__name__)

__all__ = ["Image", "save_video"]


class Image(object):

    def __init__(self, data):
        """A wrapper for a grayscale image.

        Args:
            data (2d array-like): An image data.
        """
        assert data.ndim == 2
        self.__data = data

    def as_array(self):
        return self.__data

    @property
    def size(self):
        return self.__data.size

    @property
    def shape(self):
        return self.__data.shape

    def as_8bit(self, cmin=None, cmax=None, low=None, high=None):
        """Same as scipy.misc.bytescale

        Args:
            cmin (float, optional): Defaults to the minimum in data.
            cmax (float, optional): Defaults to the maximum in data.
            low (float, optional): Defaults to 0
            high (float, optional): Defaults to 255.

        Returns:
            Image: an Image object with 8bit data.
        """
        cmin = self.__data.min() if cmin is None else cmin
        cmax = self.__data.max() if cmax is None else cmax
        low = 0 if low is None else low
        high = 255 if high is None else high

        cscale = cmax - cmin
        scale = float(high - low) / cscale
        bytedata = (self.__data - cmin) * scale + low
        bytedata = (bytedata.clip(low, high) + 0.5).astype(numpy.uint8)
        return Image(bytedata)

    def save(self, filename):
        """Save the image.

        Note:
            Requires `PIL`.

        Args:
            filename (str): An output file name.
        """
        if self.__data.dtype is not numpy.uint8:
            data = self.as_8bit().as_array()
        else:
            data = self.__data

        import PIL.Image
        img = PIL.Image.fromarray(data)
        img.save(filename)

    def plot(self):
        """Plot the image.

        Note:
            Requires `plotly` or `matplotlib`.
        """
        try:
            import plotly.express as px
        except ImportError:
            import matplotlib.pyplot as plt
            _ = plt.figure()
            plt.imshow(self.__data, interpolation='none', cmap='gray')
            plt.show()
        else:
            fig = px.imshow(self.__data, color_continuous_scale='gray')
            fig.show()

    def _ipython_display_(self):
        """
        Displays the object as a side effect.
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        self.plot()

def save_video(filename, imgs, interval=100, dpi=None):
    """Make a video from images.

    Note:
        Requires `matplotlib`.

    Args:
        filename (str): An output file name.
        imgs (list): A list of Images.
        interval (int, optional): An interval between frames given in millisecond.
        dpi (int, optional): dpi.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    def animate(frame, image, imgs):
        image.set_array(imgs[frame].as_array())
        return image

    plt.ioff()
    fig, ax = plt.subplots(1, figsize=(1, 1))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    image = ax.imshow(imgs[0].as_array(), cmap='gray')
    animation = FuncAnimation(
        fig, animate, numpy.arange(len(imgs)), fargs=(image, imgs), interval=interval)
    dpi = dpi or max(imgs[0].shape)
    animation.save(filename, dpi=dpi)
    plt.clf()
    plt.ion()
