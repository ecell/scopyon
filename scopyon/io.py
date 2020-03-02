import os.path
import csv
import contextlib
import io
import pkgutil
from collections import defaultdict

from logging import getLogger
_log = getLogger(__name__)


def isfloat(val):
    """Check if the given string can be converted to float or not.
    This method is fast when returning True

    Args:
        val (str): The input

    Returns:
        bool: Return True if the input can be converted to float

    """
    try:
        float(val)
        return True
    except ValueError:
        return False

def read_catalog(data_path):
    package_name = 'scopyon'
    data = pkgutil.get_data(package_name, data_path).decode()

    catalog_data = defaultdict(list)
    with contextlib.closing(io.StringIO(data)) as fin:
        for _ in range(5):
            line = fin.readline()
            line = line.rstrip()
            _log.debug('     {}'.format(line))

        reader = csv.reader(fin)

        row = next(reader)
        if len(row) != 1 or isfloat(row[0]):
            raise RuntimeError('A catalog in invalid format was given [{}]'.format(filename))
        key = row[0]

        for row in reader:
            if len(row) == 1 and not isfloat(row[0]):
                key = row[0]
            elif len(row) != 0:
                catalog_data[key].append(row)
    return catalog_data

def read_fluorophore_catalog(fluorophore_type):
    catalog_data = read_catalog('catalog/fluorophore/{}.csv'.format(fluorophore_type))
    return catalog_data['Excitation'], catalog_data['Emission']

def read_dichroic_catalog(dm):
    return read_catalog('catalog/dichroic/{}.csv'.format(dm))['wavedataset']

def read_excitation_catalog(excitation):
    return read_catalog('catalog/excitation/{}.csv'.format(excitation))['wavedataset']

def read_emission_catalog(emission):
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), )
    return read_catalog('catalog/emission/{}.csv'.format(emission))['wavedataset']
