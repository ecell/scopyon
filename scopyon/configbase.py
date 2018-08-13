import os
import re
import configparser
import collections
import ast

import numpy

from logging import getLogger
_log = getLogger(__name__)


class _Config:

    def __init__(self, filename=None, config=None):
        self.__data = collections.defaultdict(lambda: None)

        if filename is not None:
            self.read(filename)
        elif config is not None:
            if isinstance(config, str):
                self.read_string(config)
            elif isinstance(config, dict):
                # for key, val in config.items():
                #     self.update(key, val)

                for key, val in config.items():
                    if key.startswith('_'):
                        continue  #XXX: For compatibility
                    # elif isinstance(val, (dict, list)):
                    #     self.__update(key, copy.deepcopy(val))
                    # else:
                    #     self.__update(key, val)
                    elif isinstance(val, (dict, list)):
                        self.update(key, copy.deepcopy(val))
                    else:
                        self.update(key, val)
            else:
                raise ValueError("Argument [conf] must be either 'str' or 'dict'.")
        else:
            filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'defaults.ini')
            self.read(filename)

    def read_string(self, config):
        parser = configparser.SafeConfigParser()
        parser.optionxform = str  # Preserving case
        parser.read_string(config)
        for sec in parser.sections():
            for opt, val in parser[sec].items():
                self.__update(opt, ast.literal_eval(val))  #XXX: None is accepted here for compatibility

    def read(self, filename):
        if not os.path.isfile(filename):
            raise IOError('No such file [{}].'.format(filename))

        parser = configparser.SafeConfigParser()
        parser.optionxform = str  # Preserving case
        parser.read(filename)
        for sec in parser.sections():
            for opt, val in parser[sec].items():
                self.__update(opt, ast.literal_eval(val))  #XXX: None is accepted here for compatibility

    def write(self, filename):
        parser = configparser.SafeConfigParser()
        parser.optionxform = str  # Preserving case
        data = {}
        for opt, val in self.__data.items():
            if isinstance(val, str):
                data[opt] = "'{}'".format(re.sub(u'[\']', lambda mo: u'\\' + mo.group(), val))
            elif isinstance(val, numpy.ndarray):
                data[opt] = str(tuple(val))
            elif isinstance(val, (int, float, bool, tuple, list, dict)):
                data[opt] = str(val)
            elif val is None:
                continue
            else:
                raise RuntimeError('A value type unsupported [{}] was found [{}, {}]'.format(type(val), opt, val))
        parser.read_dict({'all': data})
        with open(filename, 'w') as f:
            parser.write(f, space_around_delimiters=True)

    def __update(self, key, val):
        _log.debug('EPIFMConfig.__update: {} = {}'.format(key, val))
        # setattr(self, key, val)
        self.__data[key] = val

    def update(self, key, val):
        if val is not None:
            self.__update(key, val)
        elif hasattr(self, key):
            _log.debug('EPIFMConfig.update: None was given for [{}]. The value was not changed [{}]'.format(key, getattr(self, key)))
        else:
            _log.debug('EPIFMConfig.update: None was given for [{}]. Ignored'.format(key))

    def __getattr__(self, name):
        return self.__data[name]

    def keys(self):
        return self.__data.keys()
