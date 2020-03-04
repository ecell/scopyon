import numpy

from logging import getLogger
_log = getLogger(__name__)

__all__ = ["Image", "save_video"]


class Image(object):

    def __init__(self, data=None, *, file=None, red=None, green=None, blue=None):
        """A wrapper for an image.

        Args:
            data (2d or 3d ndarray): An image data.
            file (str, optional): A file path.
            red (2d ndarray or Image, optional): RGB channel
            green (2d ndarray or Image, optional): RGB channel
            blue (2d ndarray or Image, optional): RGB channel
        """
        if data is not None:
            assert file is None and red is None and green is None and blue is None
            assert data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 3)
            self.__data = data
        elif file is not None:
            assert red is None and green is None and blue is None
            import matplotlib.pyplot as plt
            self.__data = plt.imread(file)
        elif red is not None or green is not None or blue is not None:
            shape, dtype = None, None
            for channel in (red, green, blue):
                if channel is None:
                    pass
                elif shape is None:
                    shape = channel.shape
                    dtype = channel.dtype
                else:
                    assert shape == channel.shape and dtype == channel.dtype
            assert shape is not None
            rgb = numpy.zeros((shape[0], shape[1], 3), dtype=dtype)
            for i, channel in enumerate((red, green, blue)):
                if isinstance(channel, Image):
                    rgb[:, :, i] = channel.as_array()
                elif isinstance(channel, numpy.ndarray):
                    rgb[:, :, i] = channel
                else:
                    assert channel is None
            self.__data = rgb
        else:
            raise ValueError("Either 'data' or 'file' must be given.")

    def as_array(self):
        return self.__data

    @property
    def dtype(self):
        return self.__data.dtype

    @property
    def ndim(self):
        return self.__data.ndim

    @property
    def size(self):
        return self.__data.size

    @property
    def shape(self):
        return self.__data.shape

    @staticmethod
    def __as_8bit(data, cmin=None, cmax=None, low=None, high=None):
        """Same as scipy.misc.bytescale

        Args:
            data (array): data
            cmin (float, optional): Defaults to the minimum in data.
            cmax (float, optional): Defaults to the maximum in data.
            low (float, optional): Defaults to 0
            high (float, optional): Defaults to 255.

        Returns:
            array: 8bit data
        """
        cmin = data.min() if cmin is None else cmin
        cmax = data.max() if cmax is None else cmax
        low = 0 if low is None else low
        high = 255 if high is None else high

        cscale = cmax - cmin
        if cscale == 0.0:
            return numpy.ones(data.shape, dtype=numpy.uint8) * low
        scale = float(high - low) / cscale
        bytedata = (data - cmin) * scale + low
        bytedata = (bytedata.clip(low, high) + 0.5).astype(numpy.uint8)
        return bytedata

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
        if self.dtype is numpy.uint8:
            data = self.__data.copy()
        elif self.ndim == 2:
            data = self.__as_8bit(
                    self.__data, cmin=cmin, cmax=cmax, low=low, high=high)
        elif self.ndim == 3:
            assert self.__data.shape[2] == 3
            data = numpy.zeros(self.__data.shape, dtype=numpy.uint8)
            for i in range(3):
                data[:, :, i] = self.__as_8bit(
                        self.__data[:, :, i], cmin=cmin, cmax=cmax, low=low, high=high)
        else:
            assert False
        return Image(data)

    def save(self, filename):
        """Save the image.

        Note:
            Requires `PIL`.

        Args:
            filename (str): An output file name.
        """
        import PIL.Image
        img = PIL.Image.fromarray(self.as_8bit().as_array())
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
            if self.ndim == 2:
                plt.imshow(self.__data, interpolation='none', cmap='gray')
            else:
                plt.imshow(self.__data, interpolation='none')
            plt.show()
        else:
            if self.ndim == 2:
                fig = px.imshow(self.__data, color_continuous_scale='gray')
            else:
                fig = px.imshow(self.__data)
            fig.show()

    def _ipython_display_(self):
        """
        Displays the object as a side effect.
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        self.plot()

def save_video(filename, imgs, interval=100, dpi=None, cmin=None, cmax=None, low=None, high=None):
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

    def animate(frame, image, imgs, cmin, cmax, low, high):
        image.set_array(imgs[frame].as_8bit(cmin=cmin, cmax=cmax, low=low, high=high).as_array())
        return image

    if cmin is None:
        cmin = min([img.as_array().min() for img in imgs])
    if cmax is None:
        cmax = max([img.as_array().max() for img in imgs])

    plt.ioff()
    fig, ax = plt.subplots(1, figsize=(1, 1))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    image = ax.imshow(imgs[0].as_8bit(cmin=cmin, cmax=cmax, low=low, high=high).as_array(), cmap='gray')
    animation = FuncAnimation(
        fig, animate, numpy.arange(len(imgs)), fargs=(image, imgs, cmin, cmax, low, high), interval=interval)
    dpi = dpi or max(imgs[0].shape)
    animation.save(filename, dpi=dpi)
    plt.clf()
    plt.ion()
