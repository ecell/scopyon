import collections.abc
import numbers
import warnings

import numpy

import scopyon._epifm
import scopyon.config
import scopyon.image

from logging import getLogger
_log = getLogger(__name__)

__all__ = ["form_image", "generate_images", "create_simulator"]


class EnvironSettings:

    def __init__(self, config):
        self.initialize(config)

    def initialize(self, config):
        self.processes = config.processes

class EPIFMSimulator(object):

    def __init__(self, config=None, method=None, rng=None):
        """ Constructor

        Args:
            config (Configuration or str, optional): Configurations.
                The default is None.
            method (str, optional): A name of method used.
                The default is None (config.default).
            rng (numpy.RandomState, optional): A random number generator.
                The default is None.
        """
        if config is None:
            config = scopyon.config.Configuration()
        elif isinstance(config, str):
            config = scopyon.config.Configuration(filename=config)
        elif not isinstance(config, scopyon.config.Configuration):
            raise TypeError("Configuration or str must be given [{}].".format(type(config)))

        if rng is None:
            warnings.warn('A random number generator is not given.')
            rng = numpy.random.RandomState()

        self.__config = config
        self.__method = method or config.default.lower()
        self.__rng = rng

    def base(self):
        return scopyon._epifm._EPIFMSimulator(
                configs=scopyon._epifm.EPIFMConfigs(self.__config[self.__method], rng=self.__rng),
                environ=EnvironSettings(self.__config.environ))

    def __format_data(self, inputs):
        assert isinstance(inputs, numpy.ndarray)

        if inputs.ndim != 2:
            raise ValueError("The given 'inputs' has wrong dimension.")

        if inputs.shape[1] == 2:
            data = numpy.hstack((
                numpy.zeros((inputs.shape[0], 1)),
                inputs * self.__config.preprocessing.scale))
            data = numpy.hstack((
                data,
                numpy.zeros((data.shape[0], 2))))
            data[:, 3] = numpy.arange(inputs.shape[0])  #FIXME: Molecule ID
            data[:, 4] = 1.0  # Photon state
        elif inputs.shape[1] == 3:
            origin = numpy.array(self.__config.preprocessing.origin)
            data = inputs * self.__config.preprocessing.scale - origin
            unit_z = numpy.cross(
                self.__config.preprocessing.unit_x,
                self.__config.preprocessing.unit_y)
            data = numpy.hstack((
                numpy.dot(data, unit_z).reshape((-1, 1)),
                numpy.dot(data, self.__config.preprocessing.unit_x).reshape((-1, 1)),
                numpy.dot(data, self.__config.preprocessing.unit_y).reshape((-1, 1)),
                ))
            data = numpy.hstack((
                data,
                numpy.zeros((data.shape[0], 2))))
            data[:, 3] = numpy.arange(inputs.shape[0])  #FIXME: Molecule ID
            data[:, 4] = 1.0  # Photon state
        elif inputs.shape[1] == 4:
            data = numpy.hstack((
                numpy.zeros((inputs.shape[0], 1)),
                inputs[:, : 2] * self.__config.preprocessing.scale,
                inputs[:, 2: ]))
        elif inputs.shape[1] == 5:
            origin = numpy.array(self.__config.preprocessing.origin)
            data = inputs[:, : 3] * self.__config.preprocessing.scale - origin
            unit_z = numpy.cross(
                self.__config.preprocessing.unit_x,
                self.__config.preprocessing.unit_y)
            data = numpy.hstack((
                numpy.dot(data, unit_z).reshape((-1, 1)),
                numpy.dot(data, self.__config.preprocessing.unit_x).reshape((-1, 1)),
                numpy.dot(data, self.__config.preprocessing.unit_y).reshape((-1, 1)),
                inputs[:, 3: ]))
        else:
            raise ValueError("The given 'inputs' has wrong shape.")
        return data

    def __format_inputs(self, inputs):
        if isinstance(inputs, numpy.ndarray):
            # Check this first. numpy.ndarray is even Iterable.
            data = ((0.0, self.__format_data(inputs)), )
        elif isinstance(inputs, collections.abc.Iterable):
            data = []
            for elem in inputs:
                if not (isinstance(elem, (tuple, list)) and len(elem) == 2
                        and isinstance(elem[0], numbers.Real) and isinstance(elem[1], numpy.ndarray)):
                    raise ValueError("The given 'inputs' has wrong type.")
                data.append((elem[0], self.__format_data(elem[1])))
        else:
            raise TypeError(
                    "Invalid argument was given [{}]."
                    " A ndarray is expected.".format(type(inputs)))
        return data

    def form_image(
            self, inputs, start_time=0.0, exposure_time=None,
            full_output=False):
        """Form image.

        Args:
            inputs (array_like): A list of points. The shape must be '(n, 3)', where
                'n' means the number of points.
            start_time (float, optional): A time to start detecting.
                Defaults to 0.0.
            exposure_time (float, optional): An exposure time.
                Defaults to `detector.exposure_time` in the configuration.
            full_output (bool, optional):
                True if to return a dictionary of optional outputs as the second output

        Returns:
            Image: An image.
            dict: only returned if full_output == True
        """
        data = self.__format_inputs(inputs)
        base = self.base()
        camera, infodict = base.output_frame(
                data, start_time=start_time, exposure_time=exposure_time, rng=self.__rng)
        # camera[:, :, 0] => expected
        # camera[:, :, 1] => ADC
        img = scopyon.image.Image(camera[:, :, 1])
        if full_output:
            infodict.update(dict(expectation=camera[:, :, 0]))
            return img, infodict
        return img

    def generate_images(
            self, inputs, num_frames, start_time=0.0, exposure_time=None,
            full_output=False):
        """Form image.

        Args:
            inputs (array_like): A list of points. The shape must be '(n, 3)', where
                'n' means the number of points.
            num_frames (int): The number of frames taken.
            start_time (float, optional): A time to start detecting.
                Defaults to `0.0` in the configuration.
            exposure_time (float, optional): An exposure time.
                Defaults to `detector_exposure_time` in the configuration.
            full_output (bool, optional):
                True if to return a dictionary of optional outputs as the second output

        Yields:
            Image: An image.
            dict: only returned if full_output == True
        """
        data = self.__format_inputs(inputs)
        base = self.base()
        for (camera, infodict) in base.generate_frames(
                data, num_frames, start_time=start_time, exposure_time=exposure_time, rng=self.__rng):
            img = scopyon.image.Image(camera[:, :, 1])
            if full_output:
                infodict.update(dict(expectation=camera[:, :, 0]))
                yield img, infodict
            else:
                yield img

def create_simulator(config=None, method=None, rng=None):
    """Return a simulator.

    Args:
        config (Configuration, optional): Configurations.
        method (str, optional): A name of method used.
            The default is None (config.default).
        rng (numpy.RandomState, optional): A random number generator.
            The default is None.

    Returns:
        EPIFMSimulator: A simulator
    """
    if method is None:
        method = config.get('method', 'default').lower()
    else:
        method = method

    simulator_type = config[method].type.lower()
    if simulator_type == 'epifm':
        return EPIFMSimulator(config=config, method=method, rng=rng)
    else:
        raise ValueError(f"An unknown type [{simulator_type}] was given.")

def form_image(
        inputs, start_time=0.0, exposure_time=None, *,
        method=None, config=None, rng=None, full_output=False):
    """Form image.

    Args:
        inputs (array_like): A list of points. The shape must be '(n, 3)',
            where 'n' means the number of points.
        start_time (float, optional): A time to start detecting.
            Defaults to 0.0.
        exposure_time (float, optional): An exposure time.
            Defaults to `detector.exposure_time` in the configuration.
        method (str, optional): A name of method used.
            The default is None ('default').
        config (Configuration, optional): Configurations.
            The default is None.
        rng (numpy.RandomState, optional): A random number generator.
            The default is None.
        full_output (bool, optional):
            True if to return a dictionary of optional outputs as the second output

    Returns:
        Image: An image
        dict: only returned if full_output == True
    """
    sim = create_simulator(config, method=method, rng=rng)
    return sim.form_image(inputs, start_time, exposure_time, full_output=full_output)

def generate_images(
        inputs, num_frames, start_time=0.0, exposure_time=None, *,
        method=None, config=None, rng=None, full_output=False):
    """Form images.

    Args:
        inputs (array_like): A list of points. The shape must be '(n, 3)',
            where 'n' means the number of points.
        num_frames (int): The number of frames taken.
        start_time (float, optional): A time to start detecting.
            Defaults to 0.0.
        exposure_time (float, optional): An exposure time.
            Defaults to `detector_exposure_time` in the configuration.
        method (str, optional): A name of method used.
            The default is None ('default').
        config (Configuration, optional): Configurations.
            The default is None.
        rng (numpy.RandomState, optional): A random number generator.
            The default is None.
        full_output (bool, optional):
            True if to return a dictionary of optional outputs as the second output

    Yields:
        Image: An image.
        dict: only returned if full_output == True
    """
    sim = create_simulator(config, method=method, rng=rng)
    return sim.generate_images(inputs, num_frames, start_time, exposure_time, full_output=full_output)


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    config = Configuration(filename)
    print(config.yaml)
    n = 10
    inputs = numpy.hstack((numpy.random.uniform(0, 5e-5, size=(n, 2)), numpy.zeros((n, 1)), ))
    print(form_image(inputs, config=config))
