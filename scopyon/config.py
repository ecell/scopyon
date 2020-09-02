import collections.abc
import contextlib
import io
import pkgutil
import warnings

import numpy

import pint
from scopyon.constants import Q_  # pint

from logging import getLogger
_log = getLogger(__name__)

__all__ = ["Configuration", "DefaultConfiguration"]


def dict_merge(dct, merge_dct):
    for k, v in merge_dct.items():
        if isinstance(dct.get(k), dict) and isinstance(v, collections.abc.Mapping):
            dict_merge(dct[k], v)
        else:
            dct[k] = v

class Configuration(collections.abc.Mapping):
    """

    Note:
        Requires `yaml` and `pint`.
    """

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
        return getattr(self, key)
        # value = self.__yaml[key]
        # if isinstance(value, dict):
        #     assert 'value' in value
        #     return value['value']
        # return value

    def __len__(self):
        return len(self.__yaml)

    def __iter__(self):
        return (key for key, value in self.__yaml.items() if not isinstance(value, dict) or 'value' in value)

    def __getattr__(self, key):
        if key not in self.__yaml:
            raise KeyError(f"'{key}'")
        value = self.__yaml[key]
        if isinstance(value, dict):
            if 'value' not in value:
                return Configuration(yaml=value)
            elif value['value'] is not None and 'units' in value:
                #XXX: value['value'] could be None (null).
                given = Q_(value['value'], value['units'])
                v = given.to_base_units()
                if given.units != v.units:
                    warnings.warn(f"Unit conversion in '{key}' from [{given.units:~}] to [{v.units:~}]")
                return v.magnitude
            else:
                return value['value']
        return value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return
        if key not in self.__yaml:
            raise KeyError(f"'{key}'")
        if isinstance(value, dict):
            raise TypeError(f"The given value for '{key}' has wrong type: {value}")

        original = self.__yaml[key]
        if isinstance(original, dict):
            if 'value' not in original:
                #TODO: value might be Configuration.
                raise ValueError(f"Cannot update '{key}'.")
        if isinstance(value, pint.Quantity) and not isinstance(value, Q_):
            # Accept a quantity with the different registry.
            warnings.warn(
                "Quantity with the different registry was given. Use scopyon.constants.Quantity")
            value = Q_(value.magnitude, value.units)
        if isinstance(value, Q_):
            if isinstance(original, dict) and 'units' in original:
                other = Q_(original['value'], original['units'])
                if not value.check(other):
                    from pint.errors import DimensionalityError
                    raise DimensionalityError(
                        value.units, other.units, value.dimensionality, other.dimensionality)
            self.__yaml[key] = dict(value=value.magnitude, units='{:~}'.format(value.units))
        else:
            self.__yaml[key] = value  #XXX: clear 'units'

class DefaultConfiguration(Configuration):
    """

    Default configuration. All the settings are available in package data, `scopyon.yaml`.
    See https://github.com/ecell/scopyon/blob/master/scopyon/scopyon.yaml
    """

    def __init__(self):
        import yaml
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        with contextlib.closing(io.StringIO(pkgutil.get_data("scopyon", "scopyon.yaml").decode())) as f:
            data = yaml.load(f.read(), Loader=Loader)

        Configuration.__init__(self, yaml=data)
