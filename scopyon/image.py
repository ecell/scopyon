import os.path
import pathlib
import numpy

from logging import getLogger
_log = getLogger(__name__)

__all__ = ["Image", "Video"]

from scopyon import _plotly, _matplotlib

class Image(object):

    PLOTTING = _plotly

    @staticmethod
    def load(file):
        """Load an image.

        Args:
            file (str): A file path.

        Returns:
            Image: an image object
        """
        assert isinstance(file, (str, pathlib.PurePath))
        filename = file if isinstance(file, str) else str(file)
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.npy':
            return Image(numpy.load(filename))
        elif ext.lower() == '.csv':
            return Image(numpy.loadtxt(filename))

        #XXX: Expected to be an image format
        import matplotlib.pyplot as plt
        return Image(plt.imread(filename))

    @staticmethod
    def RGB(red=None, green=None, blue=None):
        """Make an image from RGB arrays.

        Args:
            red (2d ndarray or Image): RGB channel
            green (2d ndarray or Image): RGB channel
            blue (2d ndarray or Image): RGB channel

        Returns:
            Image: an image object
        """
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
        return Image(rgb)

    def __init__(self, data):
        """A wrapper for an image.

        Args:
            data (2d or 3d ndarray): An image data.
        """
        assert data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 3)
        self.__data = data

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

    def save(self, filename, **kwargs):
        """Save the 8-bit image.

        Note:
            Requires `pillow` to save in the image format.
            See also `scopyon.Image.savefig`.

        Args:
            filename (str): An output file name.
                `.npy`, `.csv` or image formats are accepted.
        """
        assert isinstance(filename, (str, pathlib.PurePath))
        filename = filename if isinstance(filename, str) else str(filename)

        _, ext = os.path.splitext(filename)
        if ext.lower() == '.npy':
            assert len(kwargs) == 0  # No option is allowed
            numpy.save(filename, self.__data)
        elif ext.lower() == '.csv':
            assert len(kwargs) == 0  # No option is allowed
            numpy.savetxt(filename, self.__data)
        else:
            #XXX: Expected to be an image format
            self.savefig(filename, self.as_8bit().as_array(), **kwargs)

    @staticmethod
    def savefig(filename, img, shapes=None):
        """ Save figure.

        Args:
            filename (str): An output file path.
            img (ndarray): An image data to be shown.
            shapes (list, optional): A list of shapes.
                shape is a dictionary consisting of `x` (row), `y` (column), `sigma` and `color`.
                `sigma` is a half size of the box (square).
        """
        import PIL.Image
        from PIL import ImageDraw

        img = PIL.Image.fromarray(img)

        if shapes is not None:
            img = img.convert('RGB')
            d = ImageDraw.Draw(img)
            for shape in shapes:
                row, column, sigma, c = shape['x'], shape['y'], shape['sigma'], shape['color']
                x, y = column, row  # imshow
                d.rectangle([(x - sigma, y - sigma), (x + sigma, y + sigma)], outline=c, width=1)

        img.save(filename)
        # import matplotlib.pyplot as plt
        # plt.ioff()
        # _, ax = plt.subplots()
        # if img.ndim == 2:
        #     plt.imshow(img, interpolation='none', cmap='gray')
        # elif img.ndim == 3:
        #     plt.imshow(img, interpolation='none')
        # else:
        #     raise ValueError("'img' has wrong dimension.")
        # if shapes is not None:
        #     for shape in shapes:
        #         ax.add_patch(__get_shape(shape))
        # plt.axis('off')
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        # plt.clf()
        # plt.ion()

    def show(self, **kwargs):
        """Plot the 8-bit image.

        Note:
            Requires `plotly` optionally.
            See also `scopyon._plotly.show`, `scopyon._matplotlib.show`.

        """
        try:
            self.PLOTTING.show(self.as_8bit().as_array(), **kwargs)
        except ImportError:
            _matplotlib.show(self.as_8bit().as_array(), **kwargs)

    def _ipython_display_(self):
        """
        Displays the object as a side effect.
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        self.show()

class Video:

    @staticmethod
    def save(filename, imgs, interval=100, dpi=None, cmin=None, cmax=None, low=None, high=None):
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
