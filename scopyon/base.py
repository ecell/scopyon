from logging import getLogger
_log = getLogger(__name__)

import numpy
import collections.abc
import warnings

from scipy.special import j0, i1e
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

import scopyon.epifm

from scopyon import io
from scopyon import constants



def dict_merge(dct, merge_dct):
    for k, v in merge_dct.items():
        if isinstance(dct.get(k), dict) and isinstance(v, collections.Mapping):
            dict_merge(dct[k], v)
        else:
            dct[k] = v

class Configuration(collections.abc.Mapping):

    def __init__(self, filename=None, yaml=None):
        if filename is not None:
            assert yaml is None
            import yaml
            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader
            with open(filename) as f:
                self.__yaml = yaml.load(f.read(), Loader=Loader)
        elif yaml is not None:
            self.__yaml = yaml
        else:
            self.__yaml = None

    def __repr__(self):
        import yaml
        try:
            from yaml import CDumper as Dumper
        except ImportError:
            from yaml import Dumper
        return yaml.dump(self.__yaml, default_flow_style=False, Dumper=Dumper)

    def update(self, conf):
        if isinstance(conf, Configuration):
            dict_merge(self.__yaml, conf.yaml)
        elif isinstance(conf, dict):
            dict_merge(self.__yaml, conf)
        else:
            import yaml
            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader
            dict_merge(self.__yaml, yaml.load(conf, Loader=Loader))

    @property
    def yaml(self):
        return self.__yaml

    def get(self, key, defaultobj=None):
        return self.__yaml.get(key, defaultobj)

    def __getitem__(self, key):
        value = self.__yaml[key]
        if isinstance(value, dict):
            assert 'value' in value
            return value['value']
        return value

    def __len__(self):
        return len(self.__yaml)

    def __iter__(self):
        return (key for key, value in self.__yaml.items() if not isinstance(value, dict) or 'value' in value)

    def __getattr__(self, key):
        assert key in self.__yaml
        value = self.__yaml[key]
        if isinstance(value, dict):
            if 'value' not in value:
                return Configuration(yaml=value)
            else:
                return value['value']
        return value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return

        assert key in self.__yaml
        assert not isinstance(value, dict)
        value_ = self.__yaml[key]
        if isinstance(value_, dict):
            if 'value' not in value_:
                return Configuration(yaml=value_)
            else:
                value_['value'] = value
        self.__yaml[key] = value

class Image(object):

    def __init__(self, data):
        assert data.ndim == 2
        self.__data = data

    def as_array(self):
        return self.__data

    def save(self, filename, low=None, high=None):
        data = self.as_array()
        low = data.min() if low is None else low
        high = data.max() if high is None else high
        data = (high - data) / (high - low) * 255
        data = numpy.uint8(self.__data)

        import PIL.Image
        img = PIL.Image.fromarray(data)
        img.save(filename)

    def plot(self):
        try:
            import plotly.express as px
        except ImportError:
            import matplotlib.pylab as plt
            _ = plt.figure()
            plt.imshow(self.__data, interpolation='none', cmap="gray")
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

class EPIFMSimulator(object):

    def __init__(self, config=None):
        """ Constructor

        Args:
            config (Configuration or str, optional): Configurations.
                The default is None.
        """
        if config is None:
            config = Configuration()
        elif isinstance(config, str):
            config = Configuration(filename=config)
        elif not isinstance(config, Configuration):
            raise TypeError("Configuration or str must be given [{}].".format(type(config)))

        self.__config = config

    def base(self, rng=None):
        # return scopyon.epifm.EPIFMSimulator(self.__config, rng=rng)
        return scopyon.epifm.EPIFMSimulator(
                configs=scopyon.epifm.EPIFMConfigs(self.__config.default.epifm, rng=rng),
                effects=scopyon.epifm.PhysicalEffectConfigs(self.__config.default.effects),
                environ=scopyon.epifm.EnvironConfigs(self.__config.environ))

    def format_inputs(self, inputs):
        if isinstance(inputs, numpy.ndarray):
            if inputs.ndim != 2:
                raise ValueError("The given 'inputs' have wrong dimension.")

            if inputs.shape[1] == 2:
                data = numpy.hstack((
                    numpy.zeros((inputs.shape[0], 1)),
                    inputs))
            elif inputs.shape[1] == 3:
                origin = numpy.array(self.__config.preprocessing.origin)
                data = inputs - origin
                unit_z = numpy.cross(
                    self.__config.preprocessing.unit_x,
                    self.__config.preprocessing.unit_y)
                data = numpy.hstack((
                    numpy.dot(data, unit_z).reshape((-1, 1)),
                    numpy.dot(data, self.__config.preprocessing.unit_x).reshape((-1, 1)),
                    numpy.dot(data, self.__config.preprocessing.unit_y).reshape((-1, 1))))
            else:
                raise ValueError("The given 'inputs' have wrong shape.")

            data = numpy.hstack((
                data,
                numpy.zeros((data.shape[0], 2))))
            data[:, 3] = numpy.arange(inputs.shape[0])
            data[:, 4] = 1.0
            data = ((0.0, data), )
        else:
            raise TypeError(
                    "Invalid argument was given [{}]."
                    " A ndarray is expected.".format(type(inputs)))
        return data

    def form_image(self, inputs, rng=None, full_output=False):
        """Form image.

        Args:
            inputs (array_like): A list of points. The shape must be '(n, 3)', where
                'n' means the number of points.
            rng (numpy.RandomState, optional): A random number generator.
                The default is None.

        Returns:
            Image: An image.
        """
        data = self.format_inputs(inputs)
        base = self.base(rng)
        camera, true_data = base.output_frame(data, rng=rng)
        # camera[:, :, 0] => expected
        # camera[:, :, 1] => ADC
        img = Image(camera[:, :, 1])
        if full_output:
            infodict = dict(expectation=camera[:, :, 0], true_data=true_data)
            return img, infodict
        return img

    def form_images(self, inputs, rng=None, full_output=False):
        """Form image.

        Args:
            inputs (array_like): A list of points. The shape must be '(n, 3)', where
                'n' means the number of points.
            rng (numpy.RandomState, optional): A random number generator.
                The default is None.

        Returns:
            Image: An image.
        """
        data = self.format_inputs(inputs)
        base = self.base(rng)
        results = base.output_frames(data, data_fmt=None, true_fmt=None, image_fmt=None, rng=rng)
        imgs = [Image(result[0][:, :, 1]) for result in results]
        if full_output:
            infodict = dict(
                    expectation=[result[0][:, :, 0] for result in results],
                    true_data=[result[1] for result in results])
            return imgs, infodict
        return imgs

def create_simulator(config=None, method=None):
    """Return a simulator.

    Args:
        config (Configuration, optional): Configurations.
        method (str, optional): A name of method used.
            The default is None ('epifm').

    Returns:
        scopyon.epifm.EPIFMSimulator: A simulator
    """
    DEFAULT_METHOD = 'epifm'
    if method is None:
        if config is not None:
            method = config.get('method', DEFAULT_METHOD).lower()
        else:
            method = DEFAULT_METHOD
    if method == 'epifm':
        return EPIFMSimulator(config=config)
    else:
        raise ValueError(f"An unknown method [{method}] was given.")

def form_image(inputs, *, method=None, config=None, rng=None, full_output=False):
    """Form image.

    Args:
        inputs (array_like): A list of points. The shape must be '(n, 3)',
            where 'n' means the number of points.
        method (str, optional): A name of method used.
            The default is None ('epifm').
        config (Configuration, optional): Configurations.
            The default is None.
        rng (numpy.RandomState, optional): A random number generator.
            The default is None.

    Returns:
        Image: An image
    """
    sim = create_simulator(config, method=method)
    return sim.form_image(inputs, rng=rng, full_output=full_output)

def form_images(inputs, *, method=None, config=None, rng=None, full_output=False):
    """Form images.

    Args:
        inputs (array_like): A list of points. The shape must be '(n, 3)',
            where 'n' means the number of points.
        method (str, optional): A name of method used.
            The default is None ('epifm').
        config (Configuration, optional): Configurations.
            The default is None.
        rng (numpy.RandomState, optional): A random number generator.
            The default is None.

    Returns:
        Image: An image
    """
    sim = create_simulator(config, method=method)
    return sim.form_images(inputs, rng=rng, full_output=full_output)


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    config = Configuration(filename)
    print(config.yaml)
    n = 10
    inputs = numpy.hstack((numpy.random.uniform(0, 5e-5, size=(n, 2)), numpy.zeros((n, 1)), ))
    print(form_image(inputs, config=config))
