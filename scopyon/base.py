import collections.abc

import numpy

import scopyon.epifm
import scopyon.config
import scopyon.image

from logging import getLogger
_log = getLogger(__name__)


class EPIFMSimulator(object):

    def __init__(self, config=None, method=None):
        """ Constructor

        Args:
            config (Configuration or str, optional): Configurations.
                The default is None.
            method (str, optional): A name of method used.
                The default is None (config.default).
        """
        if config is None:
            config = scopyon.config.Configuration()
        elif isinstance(config, str):
            config = scopyon.config.Configuration(filename=config)
        elif not isinstance(config, scopyon.config.Configuration):
            raise TypeError("Configuration or str must be given [{}].".format(type(config)))

        self.__config = config
        self.__method = method or config.default.lower()

    def base(self, rng=None):
        return scopyon.epifm._EPIFMSimulator(
                configs=scopyon.epifm.EPIFMConfigs(self.__config[self.__method], rng=rng),
                environ=scopyon.epifm.EnvironConfigs(self.__config.environ))

    def __format_data(self, inputs):
        assert isinstance(inputs, numpy.ndarray)

        if inputs.ndim != 2:
            raise ValueError("The given 'inputs' has wrong dimension.")

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
            raise ValueError("The given 'inputs' has wrong shape.")

        data = numpy.hstack((
            data,
            numpy.zeros((data.shape[0], 2))))
        data[:, 3] = numpy.arange(inputs.shape[0])  #FIXME: Molecule ID
        data[:, 4] = 1.0  # Photon state
        return data

    def __format_inputs(self, inputs):
        if isinstance(inputs, numpy.ndarray):
            # Check this first. numpy.ndarray is even Iterable.
            data = ((0.0, self.__format_data(inputs)), )
        elif isinstance(inputs, collections.abc.Iterable):
            data = []
            for elem in inputs:
                if not (isinstance(elem, (tuple, list)) and len(elem) == 2
                        and isinstance(elem[0], Real) and isinstance(elem[1], numpy.ndarray)):
                    raise ValueError("The given 'inputs' has wrong type.")
                data.append((elem[0], self.__format_data(elem[1])))
        else:
            raise TypeError(
                    "Invalid argument was given [{}]."
                    " A ndarray is expected.".format(type(inputs)))
        return data

    def num_frames(self, start_time=None, end_time=None, exposure_time=None):
        """Return the number of frames within the interval given.

        Args:
            start_time (float, optional): A time to open a shutter.
                Defaults to `shutter_start_time` in the configuration.
            end_time (float, optional): A time to open a shutter.
                Defaults to `shutter_end_time` in the configuration.
            exposure_time (float, optional): An exposure time.
                Defaults to `detector_exposure_time` in the configuration.

        Returns:
            int: The number of frames available within the interval.
        """
        start_time = start_time or self.__config.shutter.start_time
        end_time = end_time or self.__config.shutter.end_time
        exposure_time = exposure_time or self.__config.detector.exposure_time
        num_frames = math.ceil((end_time - start_time) / exposure_time)
        return num_frames

    def form_image(
            self, inputs, start_time=None, exposure_time=None,
            rng=None, full_output=False):
        """Form image.

        Args:
            inputs (array_like): A list of points. The shape must be '(n, 3)', where
                'n' means the number of points.
            start_time (float, optional): A time to open a shutter.
                Defaults to `shutter.start_time` in the configuration.
            exposure_time (float, optional): An exposure time.
                Defaults to `shutter.exposure_time` in the configuration.
            rng (numpy.RandomState, optional): A random number generator.
                The default is None.
            full_output (bool, optional):
                True if to return a dictionary of optional outputs as the second output

        Returns:
            Image: An image.
            dict: only returned if full_output == True
        """
        data = self.__format_inputs(inputs)
        base = self.base(rng)
        camera, true_data = base.output_frame(
                data, start_time=start_time, exposure_time=exposure_time, rng=rng)
        # camera[:, :, 0] => expected
        # camera[:, :, 1] => ADC
        img = scopyon.image.Image(camera[:, :, 1])
        if full_output:
            infodict = dict(expectation=camera[:, :, 0], true_data=true_data)
            return img, infodict
        return img

    def form_images(
            self, inputs, start_time=None, end_time=None, exposure_time=None,
            rng=None, full_output=False):
        """Form image.

        Args:
            inputs (array_like): A list of points. The shape must be '(n, 3)', where
                'n' means the number of points.
            start_time (float, optional): A time to open a shutter.
                Defaults to `shutter_start_time` in the configuration.
            end_time (float, optional): A time to open a shutter.
                Defaults to `shutter_end_time` in the configuration.
            exposure_time (float, optional): An exposure time.
                Defaults to `detector_exposure_time` in the configuration.
            rng (numpy.RandomState, optional): A random number generator.
                The default is None.
            full_output (bool, optional):
                True if to return a dictionary of optional outputs as the second output

        Returns:
            list: A list of images.
            list: A list of dict. only returned if full_output == True
        """
        data = self.__format_inputs(inputs)
        base = self.base(rng)
        results = base.output_frames(data, data_fmt=None, true_fmt=None, image_fmt=None, rng=rng)
        imgs = [scopyon.image.Image(result[0][:, :, 1]) for result in results]
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
            The default is None (config.default).

    Returns:
        scopyon.epifm.EPIFMSimulator: A simulator
    """
    if method is None:
        method = config.get('default', 'epifm').lower()
    else:
        method = method

    simulator_type = config[method].type.lower()
    if simulator_type == 'epifm':
        return EPIFMSimulator(config=config, method=method)
    else:
        raise ValueError(f"An unknown type [{simulator_type}] was given.")

def form_image(
        inputs, start_time=None, exposure_time=None, *,
        method=None, config=None, rng=None, full_output=False):
    """Form image.

    Args:
        inputs (array_like): A list of points. The shape must be '(n, 3)',
            where 'n' means the number of points.
        start_time (float, optional): A time to open a shutter.
            Defaults to `shutter.start_time` in the configuration.
        exposure_time (float, optional): An exposure time.
            Defaults to `shutter.exposure_time` in the configuration.
        method (str, optional): A name of method used.
            The default is None (config.default).
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
    sim = create_simulator(config, method=method)
    return sim.form_image(inputs, start_time, exposure_time, rng=rng, full_output=full_output)

def form_images(
        inputs, start_time=None, end_time=None, exposure_time=None, *,
        method=None, config=None, rng=None, full_output=False):
    """Form images.

    Args:
        inputs (array_like): A list of points. The shape must be '(n, 3)',
            where 'n' means the number of points.
        start_time (float, optional): A time to open a shutter.
            Defaults to `shutter_start_time` in the configuration.
        end_time (float, optional): A time to open a shutter.
            Defaults to `shutter_end_time` in the configuration.
        exposure_time (float, optional): An exposure time.
            Defaults to `detector_exposure_time` in the configuration.
        method (str, optional): A name of method used.
            The default is None ('epifm').
        config (Configuration, optional): Configurations.
            The default is None.
        rng (numpy.RandomState, optional): A random number generator.
            The default is None.
        full_output (bool, optional):
            True if to return a dictionary of optional outputs as the second output

    Returns:
        list: A list of images.
        list: A list of dict. only returned if full_output == True
    """
    sim = create_simulator(config, method=method)
    return sim.form_images(inputs, start_time, end_time, exposure_time, rng=rng, full_output=full_output)


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    config = Configuration(filename)
    print(config.yaml)
    n = 10
    inputs = numpy.hstack((numpy.random.uniform(0, 5e-5, size=(n, 2)), numpy.zeros((n, 1)), ))
    print(form_image(inputs, config=config))
